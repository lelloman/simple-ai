#!/bin/bash
set -e

# Deploy simple-ai-runner to multiple remote hosts from a configuration file
# Usage: ./deploy-fleet.sh [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/deploy-hosts.toml"
BINARY="$PROJECT_DIR/target/release/simple-ai-runner"

# CLI options
BUILD=false
DRY_RUN=false
PARALLEL=false
CONTINUE_ON_ERROR=false
SELECTED_HOSTS=""

# Default values from config
DEFAULT_SSH_USER=""
DEFAULT_DEPLOY_DIR=""
DEFAULT_SERVICE_PORT=""

# Host arrays (populated during parsing)
declare -a HOST_NAMES
declare -a HOST_HOSTNAMES
declare -a HOST_SSH_USERS
declare -a HOST_DEPLOY_DIRS
declare -a HOST_SERVICE_PORTS
declare -a HOST_CONFIG_FILES

# Deployment results
declare -a SUCCESSFUL_HOSTS
declare -a FAILED_HOSTS
declare -a UNREACHABLE_HOSTS

usage() {
    cat <<EOF
Usage: $0 [options]

Deploy simple-ai-runner to multiple remote hosts.

Options:
  --build             Build the binary before deploying
  --hosts <list>      Deploy to specific hosts (comma-separated names)
  --parallel          Deploy to hosts in parallel
  --continue-on-error Continue deploying even if a host fails
  --dry-run           Show what would be done without making changes
  --config <file>     Use alternate config file (default: deploy-hosts.toml)
  -h, --help          Show this help message

Examples:
  $0                           Deploy to all hosts
  $0 --build                   Build and deploy to all hosts
  $0 --hosts gpu-server-01     Deploy to a single host
  $0 --hosts h1,h2 --parallel  Deploy to specific hosts in parallel
  $0 --dry-run                 Preview deployment steps
EOF
    exit 0
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --hosts)
            SELECTED_HOSTS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Copy deploy-hosts.example.toml to deploy-hosts.toml and customize it."
    exit 1
fi

# Parse TOML config file
parse_config() {
    local in_defaults=false
    local in_host=false
    local host_index=-1

    # Temporary variables for current host
    local current_name=""
    local current_hostname=""
    local current_ssh_user=""
    local current_deploy_dir=""
    local current_service_port=""
    local current_config_file=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Remove leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue

        # Check for section headers
        if [[ "$line" == "[defaults]" ]]; then
            in_defaults=true
            in_host=false
            continue
        fi

        if [[ "$line" == "[[hosts]]" ]]; then
            # Save previous host if exists
            if [[ $in_host == true && -n "$current_name" ]]; then
                HOST_NAMES+=("$current_name")
                HOST_HOSTNAMES+=("$current_hostname")
                HOST_SSH_USERS+=("${current_ssh_user:-$DEFAULT_SSH_USER}")
                HOST_DEPLOY_DIRS+=("${current_deploy_dir:-$DEFAULT_DEPLOY_DIR}")
                HOST_SERVICE_PORTS+=("${current_service_port:-$DEFAULT_SERVICE_PORT}")
                HOST_CONFIG_FILES+=("$current_config_file")
            fi

            # Reset for new host
            in_defaults=false
            in_host=true
            current_name=""
            current_hostname=""
            current_ssh_user=""
            current_deploy_dir=""
            current_service_port=""
            current_config_file=""
            continue
        fi

        # Parse key = value pairs
        if [[ "$line" =~ ^([a-z_]+)[[:space:]]*=[[:space:]]*\"?([^\"]*)\"?$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            if [[ $in_defaults == true ]]; then
                case "$key" in
                    ssh_user) DEFAULT_SSH_USER="$value" ;;
                    deploy_dir) DEFAULT_DEPLOY_DIR="$value" ;;
                    service_port) DEFAULT_SERVICE_PORT="$value" ;;
                esac
            elif [[ $in_host == true ]]; then
                case "$key" in
                    name) current_name="$value" ;;
                    hostname) current_hostname="$value" ;;
                    ssh_user) current_ssh_user="$value" ;;
                    deploy_dir) current_deploy_dir="$value" ;;
                    service_port) current_service_port="$value" ;;
                    config_file) current_config_file="$value" ;;
                esac
            fi
        fi
    done < "$CONFIG_FILE"

    # Save last host
    if [[ $in_host == true && -n "$current_name" ]]; then
        HOST_NAMES+=("$current_name")
        HOST_HOSTNAMES+=("$current_hostname")
        HOST_SSH_USERS+=("${current_ssh_user:-$DEFAULT_SSH_USER}")
        HOST_DEPLOY_DIRS+=("${current_deploy_dir:-$DEFAULT_DEPLOY_DIR}")
        HOST_SERVICE_PORTS+=("${current_service_port:-$DEFAULT_SERVICE_PORT}")
        HOST_CONFIG_FILES+=("$current_config_file")
    fi
}

# Check if host is in selected hosts list
is_host_selected() {
    local name="$1"

    # If no hosts specified, all are selected
    [[ -z "$SELECTED_HOSTS" ]] && return 0

    # Check if name is in comma-separated list
    IFS=',' read -ra selected <<< "$SELECTED_HOSTS"
    for s in "${selected[@]}"; do
        [[ "$s" == "$name" ]] && return 0
    done
    return 1
}

# Validate SSH connectivity
validate_ssh() {
    local user="$1"
    local hostname="$2"
    local name="$3"

    echo -n "  Testing SSH to $name ($user@$hostname)... "
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$user@$hostname" "exit 0" 2>/dev/null; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        return 1
    fi
}

# Deploy to a single host
deploy_host() {
    local name="$1"
    local hostname="$2"
    local ssh_user="$3"
    local deploy_dir="$4"
    local service_port="$5"
    local config_file="$6"

    local ssh_target="$ssh_user@$hostname"
    local config_path="$SCRIPT_DIR/$config_file"
    local service_file="$SCRIPT_DIR/simple-ai-runner.service"

    echo ""
    echo "=== Deploying to $name ($ssh_target) ==="

    # Validate config file exists
    if [[ ! -f "$config_path" ]]; then
        echo "Error: Config file not found: $config_path"
        return 1
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would stop service on $ssh_target"
        echo "[DRY-RUN] Would kill processes on port $service_port"
        echo "[DRY-RUN] Would upload binary to $ssh_target:$deploy_dir/simple-ai-runner"
        echo "[DRY-RUN] Would upload config from $config_path to $ssh_target:$deploy_dir/config.toml"
        echo "[DRY-RUN] Would install systemd service with deploy_dir=$deploy_dir"
        echo "[DRY-RUN] Would start and verify service"
        return 0
    fi

    # Generate customized systemd service file
    local temp_service="/tmp/simple-ai-runner-$name.service"
    sed -e "s|WorkingDirectory=.*|WorkingDirectory=$deploy_dir|" \
        -e "s|ExecStart=.*|ExecStart=$deploy_dir/simple-ai-runner|" \
        "$service_file" > "$temp_service"

    # Stop service
    echo "  Stopping service..."
    ssh "$ssh_target" "systemctl --user stop simple-ai-runner 2>/dev/null || true"

    # Kill rogue processes on port
    echo "  Killing processes on port $service_port..."
    ssh "$ssh_target" "fuser -k $service_port/tcp 2>/dev/null || true"
    sleep 1

    # Upload binary
    echo "  Uploading binary..."
    scp -q "$BINARY" "$ssh_target:$deploy_dir/simple-ai-runner-new"

    # Replace binary
    echo "  Installing binary..."
    ssh "$ssh_target" "mv $deploy_dir/simple-ai-runner-new $deploy_dir/simple-ai-runner && chmod +x $deploy_dir/simple-ai-runner"

    # Upload config
    echo "  Uploading config..."
    scp -q "$config_path" "$ssh_target:$deploy_dir/config.toml"

    # Install systemd service
    echo "  Installing systemd service..."
    ssh "$ssh_target" "mkdir -p ~/.config/systemd/user"
    scp -q "$temp_service" "$ssh_target:~/.config/systemd/user/simple-ai-runner.service"
    ssh "$ssh_target" "systemctl --user daemon-reload && systemctl --user enable simple-ai-runner"
    rm -f "$temp_service"

    # Start service
    echo "  Starting service..."
    ssh "$ssh_target" "systemctl --user start simple-ai-runner"

    # Verify
    sleep 2
    if ssh "$ssh_target" "systemctl --user is-active --quiet simple-ai-runner"; then
        echo "  SUCCESS: Service is running"
        ssh "$ssh_target" "$deploy_dir/simple-ai-runner --version" 2>/dev/null || true
        return 0
    else
        echo "  FAILED: Service did not start"
        ssh "$ssh_target" "journalctl --user -u simple-ai-runner -n 10 --no-pager" 2>/dev/null || true
        return 1
    fi
}

# Main execution
main() {
    echo "=== Simple AI Runner Fleet Deployment ==="
    echo ""

    # Parse config
    echo "Parsing config: $CONFIG_FILE"
    parse_config

    if [[ ${#HOST_NAMES[@]} -eq 0 ]]; then
        echo "Error: No hosts defined in config file"
        exit 1
    fi

    echo "Found ${#HOST_NAMES[@]} host(s) in config"

    # Build list of hosts to deploy
    declare -a DEPLOY_INDICES
    for i in "${!HOST_NAMES[@]}"; do
        if is_host_selected "${HOST_NAMES[$i]}"; then
            DEPLOY_INDICES+=("$i")
        fi
    done

    if [[ ${#DEPLOY_INDICES[@]} -eq 0 ]]; then
        echo "Error: No matching hosts found for: $SELECTED_HOSTS"
        echo "Available hosts: ${HOST_NAMES[*]}"
        exit 1
    fi

    echo "Will deploy to ${#DEPLOY_INDICES[@]} host(s): ${HOST_NAMES[*]}"
    echo ""

    # Validate SSH connectivity
    echo "Validating SSH connectivity..."
    declare -a VALID_INDICES
    for i in "${DEPLOY_INDICES[@]}"; do
        if validate_ssh "${HOST_SSH_USERS[$i]}" "${HOST_HOSTNAMES[$i]}" "${HOST_NAMES[$i]}"; then
            VALID_INDICES+=("$i")
        else
            UNREACHABLE_HOSTS+=("${HOST_NAMES[$i]}")
            if [[ "$CONTINUE_ON_ERROR" != true && "$DRY_RUN" != true ]]; then
                echo "Error: Cannot reach ${HOST_NAMES[$i]}. Use --continue-on-error to skip."
                exit 1
            fi
        fi
    done

    if [[ ${#VALID_INDICES[@]} -eq 0 ]]; then
        echo "Error: No reachable hosts"
        exit 1
    fi

    # Build binary if requested
    if [[ "$BUILD" == true ]]; then
        echo ""
        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] Would build simple-ai-runner"
        else
            echo "=== Building simple-ai-runner ==="
            cargo build --release -p inference-runner --manifest-path "$PROJECT_DIR/Cargo.toml"
        fi
    fi

    # Check binary exists
    if [[ "$DRY_RUN" != true && ! -f "$BINARY" ]]; then
        echo "Error: Binary not found: $BINARY"
        echo "Run with --build to build it first."
        exit 1
    fi

    # Deploy to hosts
    if [[ "$PARALLEL" == true ]]; then
        echo ""
        echo "Deploying in parallel..."

        declare -a PIDS
        declare -a PID_NAMES

        for i in "${VALID_INDICES[@]}"; do
            (
                deploy_host "${HOST_NAMES[$i]}" "${HOST_HOSTNAMES[$i]}" \
                    "${HOST_SSH_USERS[$i]}" "${HOST_DEPLOY_DIRS[$i]}" \
                    "${HOST_SERVICE_PORTS[$i]}" "${HOST_CONFIG_FILES[$i]}"
            ) &
            PIDS+=($!)
            PID_NAMES+=("${HOST_NAMES[$i]}")
        done

        # Wait for all and collect results
        for j in "${!PIDS[@]}"; do
            if wait "${PIDS[$j]}"; then
                SUCCESSFUL_HOSTS+=("${PID_NAMES[$j]}")
            else
                FAILED_HOSTS+=("${PID_NAMES[$j]}")
            fi
        done
    else
        # Sequential deployment
        for i in "${VALID_INDICES[@]}"; do
            if deploy_host "${HOST_NAMES[$i]}" "${HOST_HOSTNAMES[$i]}" \
                "${HOST_SSH_USERS[$i]}" "${HOST_DEPLOY_DIRS[$i]}" \
                "${HOST_SERVICE_PORTS[$i]}" "${HOST_CONFIG_FILES[$i]}"; then
                SUCCESSFUL_HOSTS+=("${HOST_NAMES[$i]}")
            else
                FAILED_HOSTS+=("${HOST_NAMES[$i]}")
                if [[ "$CONTINUE_ON_ERROR" != true && "$DRY_RUN" != true ]]; then
                    echo ""
                    echo "Deployment failed. Use --continue-on-error to continue with remaining hosts."
                    break
                fi
            fi
        done
    fi

    # Print summary
    echo ""
    echo "========================================="
    echo "           DEPLOYMENT SUMMARY"
    echo "========================================="

    if [[ ${#SUCCESSFUL_HOSTS[@]} -gt 0 ]]; then
        echo "Successful (${#SUCCESSFUL_HOSTS[@]}): ${SUCCESSFUL_HOSTS[*]}"
    fi

    if [[ ${#FAILED_HOSTS[@]} -gt 0 ]]; then
        echo "Failed (${#FAILED_HOSTS[@]}): ${FAILED_HOSTS[*]}"
    fi

    if [[ ${#UNREACHABLE_HOSTS[@]} -gt 0 ]]; then
        echo "Unreachable (${#UNREACHABLE_HOSTS[@]}): ${UNREACHABLE_HOSTS[*]}"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "(This was a dry run - no changes were made)"
    fi

    # Exit with error if any failures
    if [[ ${#FAILED_HOSTS[@]} -gt 0 || ${#UNREACHABLE_HOSTS[@]} -gt 0 ]]; then
        exit 1
    fi
}

main
