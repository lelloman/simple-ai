//! ARP cache lookup for MAC address discovery.

use std::net::IpAddr;
use std::process::Command;

/// Look up MAC address for an IP from the system ARP cache.
///
/// Returns MAC in format "AA:BB:CC:DD:EE:FF" or None if not found.
pub fn lookup_mac(ip: &IpAddr) -> Option<String> {
    // Try /proc/net/arp first (Linux)
    if let Some(mac) = lookup_from_proc_arp(ip) {
        return Some(mac);
    }

    // Fall back to arp command
    lookup_from_arp_command(ip)
}

/// Look up MAC from /proc/net/arp (Linux-specific).
fn lookup_from_proc_arp(ip: &IpAddr) -> Option<String> {
    let content = std::fs::read_to_string("/proc/net/arp").ok()?;
    let ip_str = ip.to_string();

    // Format: IP address       HW type     Flags       HW address            Mask     Device
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 && parts[0] == ip_str {
            let mac = parts[3];
            // Skip incomplete entries (00:00:00:00:00:00)
            if mac != "00:00:00:00:00:00" {
                return Some(mac.to_uppercase());
            }
        }
    }
    None
}

/// Look up MAC using the arp command (cross-platform fallback).
fn lookup_from_arp_command(ip: &IpAddr) -> Option<String> {
    let output = Command::new("arp")
        .arg("-n")
        .arg(ip.to_string())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse output - format varies by OS but MAC is usually recognizable
    // Linux: "192.168.1.1  ether  aa:bb:cc:dd:ee:ff  C  eth0"
    // macOS: "? (192.168.1.1) at aa:bb:cc:dd:ee:ff on en0"
    for line in stdout.lines() {
        // Look for MAC address pattern (xx:xx:xx:xx:xx:xx)
        for word in line.split_whitespace() {
            if is_mac_address(word) && word != "00:00:00:00:00:00" {
                return Some(word.to_uppercase());
            }
        }
    }
    None
}

/// Check if a string looks like a MAC address.
fn is_mac_address(s: &str) -> bool {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 6 {
        return false;
    }
    parts.iter().all(|p| p.len() == 2 && p.chars().all(|c| c.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mac_address_valid() {
        assert!(is_mac_address("aa:bb:cc:dd:ee:ff"));
        assert!(is_mac_address("AA:BB:CC:DD:EE:FF"));
        assert!(is_mac_address("00:11:22:33:44:55"));
    }

    #[test]
    fn test_is_mac_address_invalid() {
        assert!(!is_mac_address("aa:bb:cc:dd:ee"));  // Too short
        assert!(!is_mac_address("aa:bb:cc:dd:ee:ff:00"));  // Too long
        assert!(!is_mac_address("aa-bb-cc-dd-ee-ff"));  // Wrong delimiter
        assert!(!is_mac_address("not a mac"));
        assert!(!is_mac_address(""));
    }

    #[test]
    fn test_lookup_mac_loopback() {
        // Loopback shouldn't have an ARP entry
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        let result = lookup_mac(&ip);
        assert!(result.is_none());
    }
}
