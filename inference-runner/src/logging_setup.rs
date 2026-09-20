//! Preserve application output and log-facade policies while sharing installation.
use simple_server::logging::{self, AnsiMode, LogFormat, LogOutput, LoggingOptions};
use tracing_subscriber::EnvFilter;

pub(crate) fn init(
    filter: EnvFilter,
    json: bool,
    stderr: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut options = LoggingOptions::new(filter.to_string());
    // EnvFilter accepts an empty directive set; the shared API requires explicit off.
    if options.filter.is_empty() {
        options.filter = "off".into();
    }
    options.format = if json {
        LogFormat::Json
    } else {
        LogFormat::Text
    };
    options.output = if stderr {
        LogOutput::Stderr
    } else {
        LogOutput::Stdout
    };
    options.ansi = if json || std::env::var("NO_COLOR").is_ok_and(|value| !value.is_empty()) {
        AnsiMode::Never
    } else {
        AnsiMode::Always
    };
    logging::try_init(options)?;
    tracing_log::LogTracer::builder()
        .with_max_level(tracing_log::AsLog::as_log(
            &tracing::level_filters::LevelFilter::current(),
        ))
        .init()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn child() {
        let Ok(mode) = std::env::var("LOGGING_COMPARISON_CHILD") else {
            return;
        };
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let json = std::env::var("LOGGING_JSON").unwrap() == "true";
        let stderr = std::env::var("LOGGING_STDERR").unwrap() == "true";
        if mode == "shared" {
            init(filter, json, stderr).unwrap();
        } else {
            let writer = if stderr {
                tracing_subscriber::fmt::writer::BoxMakeWriter::new(std::io::stderr)
            } else {
                tracing_subscriber::fmt::writer::BoxMakeWriter::new(std::io::stdout)
            };
            let builder = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(writer);
            if json {
                builder.json().init();
            } else {
                builder.init();
            }
        }
        tracing::debug!(target: "application", count = 2, "canary-debug");
        tracing::info!(target: "other", "canary-info");
        tracing::warn!(target: "other", "canary-warn");
        tracing::error!(target: "other", "canary-error");
        tracing_log::log::warn!(target: "dependency", "canary-log-bridge");
        let span = tracing::info_span!(target: "application", "scope", tenant = 7);
        let _entered = span.enter();
        tracing::trace!(target: "application", ready = true, "canary-scoped-trace");
        tracing::info!(target: "application", "canary-scoped-info");
    }

    fn output(
        mode: &str,
        filter: Option<&str>,
        no_color: bool,
        json: bool,
        stderr: bool,
    ) -> (Vec<String>, Vec<String>) {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "logging_setup::tests::child", "--nocapture"])
            .env("LOGGING_COMPARISON_CHILD", mode)
            .env("LOGGING_JSON", json.to_string())
            .env("LOGGING_STDERR", stderr.to_string());
        if let Some(filter) = filter {
            command.env("RUST_LOG", filter);
        } else {
            command.env_remove("RUST_LOG");
        }
        if no_color {
            command.env("NO_COLOR", "1");
        } else {
            command.env_remove("NO_COLOR");
        }
        let result = command.output().unwrap();
        assert!(result.status.success(), "{result:?}");
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
        let normalize = |bytes: Vec<u8>| {
            String::from_utf8(bytes)
                .unwrap()
                .lines()
                .filter(|line| line.contains("canary-"))
                .map(|line| {
                    if json {
                        let mut event: serde_json::Value = serde_json::from_str(line).unwrap();
                        event.as_object_mut().unwrap().remove("timestamp");
                        event.to_string()
                    } else {
                        assert_eq!(line.contains("\x1b["), !no_color);
                        line.split_once(' ').unwrap().1.to_owned()
                    }
                })
                .collect::<Vec<_>>()
        };
        (normalize(result.stdout), normalize(result.stderr))
    }

    #[test]
    fn preserves_logging_contract() {
        for filter in [
            None,
            Some(""),
            Some("   "),
            Some("off"),
            Some("warn"),
            Some("application=debug"),
            Some("application=debug,broken["),
            Some("application[scope{tenant=7}]=trace,off"),
            Some("application=off,other=warn"),
        ] {
            for no_color in [true, false] {
                for json in [true, false] {
                    for stderr in [true, false] {
                        assert_eq!(
                            output("shared", filter, no_color, json, stderr),
                            output("legacy", filter, no_color, json, stderr),
                            "filter={filter:?} color={no_color} json={json} stderr={stderr}"
                        );
                    }
                }
            }
        }
    }
}
