//! Wake-on-LAN (WOL) implementation.
//!
//! Sends magic packets to wake up offline machines.
//! Supports direct UDP broadcast or sending via a bouncer service (for Docker).

use std::net::UdpSocket;

/// Errors that can occur during WOL operations.
#[derive(Debug, thiserror::Error)]
pub enum WolError {
    #[error("Invalid MAC address format: {0}")]
    InvalidMacAddress(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Bouncer error: {0}")]
    BouncerError(String),
}

/// Parse a MAC address string (AA:BB:CC:DD:EE:FF) into bytes.
fn parse_mac_address(mac: &str) -> Result<[u8; 6], WolError> {
    let parts: Vec<&str> = mac.split(':').collect();
    if parts.len() != 6 {
        return Err(WolError::InvalidMacAddress(format!(
            "Expected 6 octets separated by ':', got {}",
            parts.len()
        )));
    }

    let mut bytes = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        bytes[i] = u8::from_str_radix(part, 16).map_err(|_| {
            WolError::InvalidMacAddress(format!("Invalid hex octet: {}", part))
        })?;
    }

    Ok(bytes)
}

/// Build a Wake-on-LAN magic packet.
///
/// The magic packet consists of:
/// - 6 bytes of 0xFF
/// - 16 repetitions of the target MAC address (96 bytes)
///
/// Total: 102 bytes
fn build_magic_packet(mac_bytes: &[u8; 6]) -> [u8; 102] {
    let mut packet = [0u8; 102];

    // First 6 bytes are 0xFF
    for byte in packet.iter_mut().take(6) {
        *byte = 0xFF;
    }

    // Repeat MAC address 16 times
    for i in 0..16 {
        let offset = 6 + (i * 6);
        packet[offset..offset + 6].copy_from_slice(mac_bytes);
    }

    packet
}

/// Send a Wake-on-LAN magic packet to wake a machine.
///
/// # Arguments
/// * `mac_address` - Target MAC address in format AA:BB:CC:DD:EE:FF
/// * `broadcast_addr` - Broadcast address to send to (e.g., "255.255.255.255")
/// * `port` - UDP port (typically 9 or 7)
///
/// # Example
/// ```ignore
/// send_wol("AA:BB:CC:DD:EE:FF", "255.255.255.255", 9)?;
/// ```
pub fn send_wol(mac_address: &str, broadcast_addr: &str, port: u16) -> Result<(), WolError> {
    let mac_bytes = parse_mac_address(mac_address)?;
    let packet = build_magic_packet(&mac_bytes);

    let socket = UdpSocket::bind("0.0.0.0:0")
        .map_err(|e| WolError::NetworkError(format!("Failed to bind socket: {}", e)))?;

    socket
        .set_broadcast(true)
        .map_err(|e| WolError::NetworkError(format!("Failed to enable broadcast: {}", e)))?;

    let dest = format!("{}:{}", broadcast_addr, port);
    socket
        .send_to(&packet, &dest)
        .map_err(|e| WolError::NetworkError(format!("Failed to send packet: {}", e)))?;

    tracing::info!(
        "Sent WOL magic packet for MAC {} to {}:{}",
        mac_address,
        broadcast_addr,
        port
    );

    Ok(())
}

/// Send WOL via bouncer service (TCP).
///
/// # Arguments
/// * `bouncer_addr` - Address of the bouncer service (e.g., "localhost:9999")
/// * `mac_address` - Target MAC address in format AA:BB:CC:DD:EE:FF
pub async fn send_wol_via_bouncer(
    bouncer_addr: &str,
    mac_address: &str,
    _broadcast_addr: &str,
) -> Result<(), WolError> {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;

    // Strip protocol prefix if present
    let addr = bouncer_addr
        .trim_start_matches("tcp://")
        .trim_start_matches("http://");

    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| WolError::BouncerError(format!("Failed to connect to bouncer at {}: {}", addr, e)))?;

    stream
        .write_all(format!("{}\n", mac_address).as_bytes())
        .await
        .map_err(|e| WolError::BouncerError(format!("Failed to send MAC to bouncer: {}", e)))?;

    tracing::info!("Sent WOL via bouncer for MAC {}", mac_address);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mac_address_valid() {
        let mac = parse_mac_address("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_lowercase() {
        let mac = parse_mac_address("aa:bb:cc:dd:ee:ff").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_mixed_case() {
        let mac = parse_mac_address("Aa:Bb:Cc:Dd:Ee:Ff").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_parse_mac_address_zeros() {
        let mac = parse_mac_address("00:00:00:00:00:00").unwrap();
        assert_eq!(mac, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_parse_mac_address_invalid_too_short() {
        let result = parse_mac_address("AA:BB:CC:DD:EE");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_invalid_too_long() {
        let result = parse_mac_address("AA:BB:CC:DD:EE:FF:00");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_invalid_hex() {
        let result = parse_mac_address("GG:BB:CC:DD:EE:FF");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_parse_mac_address_wrong_delimiter() {
        let result = parse_mac_address("AA-BB-CC-DD-EE-FF");
        assert!(matches!(result, Err(WolError::InvalidMacAddress(_))));
    }

    #[test]
    fn test_build_magic_packet() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let packet = build_magic_packet(&mac);

        // Check length
        assert_eq!(packet.len(), 102);

        // Check first 6 bytes are 0xFF
        assert_eq!(&packet[0..6], &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

        // Check MAC is repeated 16 times
        for i in 0..16 {
            let offset = 6 + (i * 6);
            assert_eq!(&packet[offset..offset + 6], &mac);
        }
    }

    #[test]
    fn test_wol_error_display() {
        let err = WolError::InvalidMacAddress("test".to_string());
        assert!(err.to_string().contains("Invalid MAC address"));

        let err = WolError::NetworkError("test".to_string());
        assert!(err.to_string().contains("Network error"));
    }
}
