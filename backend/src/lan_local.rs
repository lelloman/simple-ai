//! Temporary, administrator-controlled access from an explicit private network.
use std::net::{IpAddr, SocketAddr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use simple_server::web::http::HeaderMap;
use chrono::{DateTime, Utc};
use ipnet::IpNet;
use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct LanLocalAccess(Mutex<Option<Grant>>);

struct Grant {
    network: IpNet,
    deadline: Instant,
    expires_at: DateTime<Utc>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanLocalUpdate {
    pub enabled: bool,
    pub network: Option<String>,
    pub duration_seconds: Option<u64>,
}

#[derive(Serialize)]
pub struct LanLocalStatus {
    pub enabled: bool,
    pub network: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub remaining_seconds: u64,
}

impl LanLocalAccess {
    pub fn update(&self, update: LanLocalUpdate) -> Result<LanLocalStatus, String> {
        let grant = if update.enabled {
            let network: IpNet = update
                .network
                .as_deref()
                .ok_or("network is required")?
                .parse()
                .map_err(|_| {
                    "network must be an IPv4 or IPv6 CIDR (use /32 or /128 for one device)"
                })?;
            // Restrict the entire requested subnet, not just its first address.
            let private_ranges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"];
            if !private_ranges
                .iter()
                .any(|range| range.parse::<IpNet>().unwrap().contains(&network))
            {
                return Err("network must be within a private LAN range".into());
            }
            let seconds = update
                .duration_seconds
                .ok_or("duration_seconds is required")?;
            if !(1..=86400).contains(&seconds) {
                return Err("duration_seconds must be between 1 and 86400 (24 hours)".into());
            }
            Some(Grant {
                network: network.trunc(),
                deadline: Instant::now() + Duration::from_secs(seconds),
                expires_at: Utc::now() + chrono::Duration::seconds(seconds as i64),
            })
        } else {
            None
        };
        *self.0.lock().unwrap() = grant;
        Ok(self.status())
    }

    pub fn status(&self) -> LanLocalStatus {
        let guard = self.0.lock().unwrap();
        match guard
            .as_ref()
            .filter(|grant| grant.deadline > Instant::now())
        {
            Some(grant) => LanLocalStatus {
                enabled: true,
                network: Some(grant.network.to_string()),
                expires_at: Some(grant.expires_at),
                remaining_seconds: grant
                    .deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs()
                    .saturating_add(1),
            },
            None => LanLocalStatus {
                enabled: false,
                network: None,
                expires_at: None,
                remaining_seconds: 0,
            },
        }
    }

    pub fn allows(&self, headers: &HeaderMap, peer: Option<SocketAddr>) -> bool {
        // Never authenticate from client-controlled forwarding headers. Reject
        // forwarded requests altogether so a LAN reverse proxy cannot confer
        // LAN privileges on an external caller.
        if ["forwarded", "x-forwarded-for", "x-real-ip"]
            .iter()
            .any(|name| headers.contains_key(*name))
        {
            return false;
        }
        let Some(peer) = peer else { return false };
        let ip = match peer.ip() {
            IpAddr::V6(ip) => ip
                .to_ipv4_mapped()
                .map(IpAddr::V4)
                .unwrap_or(IpAddr::V6(ip)),
            ip => ip,
        };
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|grant| grant.deadline > Instant::now() && grant.network.contains(&ip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enable(access: &LanLocalAccess, network: &str) -> Result<LanLocalStatus, String> {
        access.update(LanLocalUpdate {
            enabled: true,
            network: Some(network.into()),
            duration_seconds: Some(60),
        })
    }

    #[test]
    fn scope_expiration_and_disable() {
        let access = LanLocalAccess::default();
        let peer = Some("192.168.1.12:1234".parse().unwrap());
        let headers = HeaderMap::new();
        assert!(!access.allows(&headers, peer));
        enable(&access, "192.168.1.0/24").unwrap();
        assert!(access.allows(&headers, peer));
        assert!(!access.allows(&headers, Some("192.168.2.12:1234".parse().unwrap())));
        assert!(!access.allows(&headers, None));
        assert!(access.allows(
            &headers,
            Some("[::ffff:192.168.1.12]:1234".parse().unwrap())
        ));
        access.0.lock().unwrap().as_mut().unwrap().deadline = Instant::now();
        assert!(!access.allows(&headers, peer));
        assert!(!access.status().enabled);
        enable(&access, "192.168.1.12/32").unwrap();
        assert!(access.allows(&headers, peer));
        assert!(!access.allows(&headers, Some("192.168.1.13:1234".parse().unwrap())));
        access
            .update(LanLocalUpdate {
                enabled: false,
                network: None,
                duration_seconds: None,
            })
            .unwrap();
        assert!(!access.allows(&headers, peer));
    }

    #[test]
    fn rejects_forwarding_and_unsafe_scopes() {
        let access = LanLocalAccess::default();
        for network in [
            "0.0.0.0/0",
            "192.168.0.0/8",
            "8.8.8.8/32",
            "127.0.0.1/32",
            "::/0",
            "invalid",
        ] {
            assert!(enable(&access, network).is_err(), "{network}");
        }
        enable(&access, "fd00::/64").unwrap();
        assert!(access.allows(&HeaderMap::new(), Some("[fd00::123]:80".parse().unwrap())));
        enable(&access, "192.168.1.0/24").unwrap();
        for name in ["forwarded", "x-forwarded-for", "x-real-ip"] {
            let mut headers = HeaderMap::new();
            headers.insert(name, "192.168.1.12".parse().unwrap());
            assert!(!access.allows(&headers, Some("192.168.1.1:80".parse().unwrap())));
        }
        for seconds in [0, 86401, u64::MAX] {
            assert!(access
                .update(LanLocalUpdate {
                    enabled: true,
                    network: Some("10.0.0.0/24".into()),
                    duration_seconds: Some(seconds)
                })
                .is_err());
        }
    }
}
