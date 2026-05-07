use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use mailparse::MailHeaderMap;
use tracing::{info, warn};

use crate::error::TermiError;

pub struct ParsedEmail {
    pub message_id: String,
    pub from: String,
    pub subject: String,
    pub date: String,
    pub body: String,
}

pub struct ImapFetcher {
    pub username: String,
    pub password: String,
    pub host: String,
}

impl ImapFetcher {
    pub fn new(username: String, password: String, host: String) -> Self {
        Self { username, password, host }
    }

    pub async fn fetch_new(
        &self,
        seen_ids: HashSet<String>,
        limit: usize,
    ) -> Result<Vec<ParsedEmail>, TermiError> {
        let username = self.username.clone();
        let password = self.password.clone();
        let host = self.host.clone();

        tokio::task::spawn_blocking(move || fetch_blocking(username, password, host, seen_ids, limit))
            .await
            .map_err(|e| TermiError::Pipeline(format!("spawn_blocking: {e}")))?
    }
}

fn fetch_blocking(
    username: String,
    password: String,
    host: String,
    seen_ids: HashSet<String>,
    limit: usize,
) -> Result<Vec<ParsedEmail>, TermiError> {
    let tls = native_tls::TlsConnector::builder()
        .build()
        .map_err(|e| TermiError::Pipeline(format!("TLS build: {e}")))?;

    let client =
        imap::connect((host.as_str(), 993u16), host.as_str(), &tls)
            .map_err(|e| TermiError::Pipeline(format!("IMAP connect to {host}: {e}")))?;

    let mut session = client
        .login(&username, &password)
        .map_err(|(e, _)| TermiError::Pipeline(format!("IMAP login: {e}")))?;

    let mailbox = session
        .select("INBOX")
        .map_err(|e| TermiError::Pipeline(format!("IMAP select INBOX: {e}")))?;

    let exists = mailbox.exists;
    info!("INBOX: {} messages total", exists);

    if exists == 0 {
        let _ = session.logout();
        return Ok(vec![]);
    }

    // Fetch the last `limit` messages by sequence number (newest are highest seq nums)
    let start = if exists > limit as u32 {
        exists - limit as u32 + 1
    } else {
        1
    };
    let seq_set = format!("{}:{}", start, exists);

    // First pass: fetch headers only to find new messages
    let header_fetches = session
        .fetch(&seq_set, "RFC822.HEADER")
        .map_err(|e| TermiError::Pipeline(format!("IMAP fetch headers: {e}")))?;

    let mut new_seqs: Vec<u32> = vec![];
    for item in header_fetches.iter() {
        if let Some(raw_h) = item.header() {
            let msg_id = extract_message_id(raw_h);
            if !seen_ids.contains(&msg_id) {
                new_seqs.push(item.message);
            }
        }
    }

    if new_seqs.is_empty() {
        info!("No new emails (all already seen)");
        let _ = session.logout();
        return Ok(vec![]);
    }

    info!("Fetching {} new emails", new_seqs.len());

    let seq_str = new_seqs
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let full_fetches = session
        .fetch(&seq_str, "RFC822")
        .map_err(|e| TermiError::Pipeline(format!("IMAP fetch RFC822: {e}")))?;

    let mut emails = vec![];
    for item in full_fetches.iter() {
        if let Some(raw) = item.body() {
            match parse_raw_email(raw) {
                Ok(email) => emails.push(email),
                Err(e) => warn!("Failed to parse email seq {}: {e}", item.message),
            }
        }
    }

    let _ = session.logout();
    Ok(emails)
}

fn extract_message_id(raw_headers: &[u8]) -> String {
    if let Ok((headers, _)) = mailparse::parse_headers(raw_headers) {
        if let Some(mid) = headers.get_first_value("Message-ID") {
            let trimmed = mid.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    format!("unknown-{}", hash_bytes(raw_headers))
}

fn hash_bytes(data: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

fn parse_raw_email(raw: &[u8]) -> Result<ParsedEmail, TermiError> {
    let parsed =
        mailparse::parse_mail(raw).map_err(|e| TermiError::Pipeline(format!("mailparse: {e}")))?;

    let message_id = parsed
        .headers
        .get_first_value("Message-ID")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("unknown-{}", hash_bytes(raw)));

    let from = parsed
        .headers
        .get_first_value("From")
        .unwrap_or_default();

    let subject = parsed
        .headers
        .get_first_value("Subject")
        .unwrap_or_default();

    let date = parsed
        .headers
        .get_first_value("Date")
        .unwrap_or_default();

    let body = extract_body(&parsed);

    Ok(ParsedEmail {
        message_id,
        from,
        subject,
        date,
        body,
    })
}

fn extract_body(mail: &mailparse::ParsedMail) -> String {
    if mail.subparts.is_empty() {
        return mail
            .get_body()
            .unwrap_or_default()
            .chars()
            .take(2000)
            .collect();
    }

    // Prefer text/plain
    for part in &mail.subparts {
        if part.ctype.mimetype == "text/plain" {
            return part
                .get_body()
                .unwrap_or_default()
                .chars()
                .take(2000)
                .collect();
        }
    }

    // Fallback: text/html stripped to markdown
    for part in &mail.subparts {
        if part.ctype.mimetype == "text/html" {
            let html = part.get_body().unwrap_or_default();
            let md = htmd::convert(&html).unwrap_or(html);
            return md.chars().take(2000).collect();
        }
    }

    // Recurse into nested multipart
    for part in &mail.subparts {
        let body = extract_body(part);
        if !body.is_empty() {
            return body;
        }
    }

    String::new()
}
