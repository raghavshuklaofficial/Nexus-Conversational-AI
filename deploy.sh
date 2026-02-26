#!/usr/bin/env bash
# ============================================================
# Nexus Conversational AI — Production Deployment Script
# ============================================================
# This script handles the full deployment lifecycle:
#   1. System prerequisites (Docker, Docker Compose)
#   2. SSL certificate issuance via Let's Encrypt
#   3. Building and starting all services
#   4. Setting up automatic certificate renewal
#
# Usage:
#   chmod +x deploy.sh
#   sudo ./deploy.sh --domain yourdomain.com --email you@email.com
# ============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse arguments ─────────────────────────────────────────
DOMAIN=""
EMAIL=""
SKIP_SSL=false
STAGING_SSL=false   # Use Let's Encrypt staging for testing

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)  DOMAIN="$2";  shift 2 ;;
        --email)   EMAIL="$2";   shift 2 ;;
        --skip-ssl) SKIP_SSL=true; shift ;;
        --staging-ssl) STAGING_SSL=true; shift ;;
        -h|--help)
            echo "Usage: sudo $0 --domain yourdomain.com --email you@email.com"
            echo ""
            echo "Options:"
            echo "  --domain       Your domain name (required)"
            echo "  --email        Email for Let's Encrypt notifications (required)"
            echo "  --skip-ssl     Skip SSL setup (HTTP only, for testing)"
            echo "  --staging-ssl  Use Let's Encrypt staging (for testing, avoids rate limits)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *) err "Unknown option: $1" ;;
    esac
done

[[ -z "$DOMAIN" ]] && err "Missing required --domain flag.  Run with --help for usage."
[[ -z "$EMAIL"  ]] && err "Missing required --email flag.   Run with --help for usage."

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Step 1: Check prerequisites ─────────────────────────────
info "Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    ok "Docker installed"
else
    ok "Docker found: $(docker --version)"
fi

if ! docker compose version &>/dev/null; then
    err "Docker Compose V2 plugin not found. Install it: https://docs.docker.com/compose/install/"
fi
ok "Docker Compose found: $(docker compose version --short)"

# ── Step 2: Create .env if missing ───────────────────────────
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    info "Creating .env from .env.example..."
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        # Generate a random Grafana password
        GF_PASS=$(openssl rand -base64 16)
        sed -i "s/^GF_ADMIN_PASSWORD=.*/GF_ADMIN_PASSWORD=${GF_PASS}/" "$PROJECT_DIR/.env"
        ok ".env created (Grafana password auto-generated)"
    else
        echo "GF_ADMIN_USER=admin" > "$PROJECT_DIR/.env"
        echo "GF_ADMIN_PASSWORD=$(openssl rand -base64 16)" >> "$PROJECT_DIR/.env"
        ok ".env created"
    fi
fi

# ── Step 3: Update nginx server_name ─────────────────────────
info "Configuring Nginx for domain: $DOMAIN"
sed -i "s/server_name _;/server_name ${DOMAIN};/g" "$PROJECT_DIR/nginx/nginx.conf"
sed -i "s/server_name _;/server_name ${DOMAIN};/g" "$PROJECT_DIR/nginx/nginx.initial.conf"
ok "Nginx configured for $DOMAIN"

# ── Step 4: Build the application ────────────────────────────
info "Building Docker images..."
cd "$PROJECT_DIR"
docker compose build --no-cache nexus-api
ok "Docker images built"

# ── Step 5: SSL Certificate Setup ───────────────────────────
if [[ "$SKIP_SSL" == true ]]; then
    warn "Skipping SSL setup (--skip-ssl). Site will run on HTTP only."
    # Use the initial (HTTP-only) nginx config
    cp "$PROJECT_DIR/nginx/nginx.initial.conf" "$PROJECT_DIR/nginx/nginx.active.conf"

    # Override the nginx volume mount to use the HTTP-only config
    NGINX_CONF="$PROJECT_DIR/nginx/nginx.initial.conf"
    docker compose up -d nexus-api redis
    docker compose run --rm -d \
        -v "$NGINX_CONF:/etc/nginx/nginx.conf:ro" \
        nginx
    ok "Services started (HTTP only)"
else
    info "Setting up SSL certificates with Let's Encrypt..."

    # 5a. Start nginx with HTTP-only config for ACME challenge
    info "Starting Nginx (HTTP-only) for domain verification..."
    docker compose up -d redis nexus-api

    # Run nginx with the initial config temporarily
    docker run -d --name nexus-nginx-init \
        --network nexus-conversational-ai_nexus-network \
        -p 80:80 \
        -v "$PROJECT_DIR/nginx/nginx.initial.conf:/etc/nginx/nginx.conf:ro" \
        -v "$PROJECT_DIR/frontend:/usr/share/nginx/html:ro" \
        -v nexus-conversational-ai_certbot-webroot:/var/www/certbot \
        nginx:alpine
    
    sleep 3
    ok "Nginx (HTTP) started for ACME challenge"

    # 5b. Request certificates
    STAGING_FLAG=""
    if [[ "$STAGING_SSL" == true ]]; then
        STAGING_FLAG="--staging"
        warn "Using Let's Encrypt STAGING environment (certificates will NOT be trusted by browsers)"
    fi

    info "Requesting SSL certificate for $DOMAIN..."
    docker run --rm \
        -v nexus-conversational-ai_certbot-webroot:/var/www/certbot \
        -v nexus-conversational-ai_certbot-certs:/etc/letsencrypt \
        certbot/certbot certonly \
            --webroot \
            --webroot-path=/var/www/certbot \
            --email "$EMAIL" \
            --agree-tos \
            --no-eff-email \
            --force-renewal \
            -d "$DOMAIN" \
            $STAGING_FLAG

    ok "SSL certificate obtained!"

    # 5c. Create symlinks so nginx finds certs at the expected path
    info "Linking certificates..."
    docker run --rm \
        -v nexus-conversational-ai_certbot-certs:/etc/letsencrypt \
        alpine sh -c "
            mkdir -p /etc/letsencrypt/live-nginx && \
            ln -sf /etc/letsencrypt/live/${DOMAIN}/fullchain.pem /etc/letsencrypt/fullchain.pem && \
            ln -sf /etc/letsencrypt/live/${DOMAIN}/privkey.pem   /etc/letsencrypt/privkey.pem
        "
    ok "Certificate symlinks created"

    # 5d. Stop temporary nginx, start full stack with HTTPS
    info "Switching to HTTPS configuration..."
    docker stop nexus-nginx-init && docker rm nexus-nginx-init

    docker compose up -d
    ok "All services started with HTTPS!"
fi

# ── Step 6: Set up auto-renewal cron ────────────────────────
if [[ "$SKIP_SSL" != true ]]; then
    info "Setting up automatic certificate renewal..."
    CRON_CMD="0 3 * * * cd $PROJECT_DIR && docker compose run --rm certbot renew --quiet && docker compose exec nginx nginx -s reload"
    (crontab -l 2>/dev/null | grep -v "certbot renew"; echo "$CRON_CMD") | crontab -
    ok "Auto-renewal cron job installed (runs daily at 3 AM)"
fi

# ── Step 7: Configure firewall ──────────────────────────────
if command -v ufw &>/dev/null; then
    info "Configuring UFW firewall..."
    ufw allow 80/tcp   >/dev/null 2>&1 || true
    ufw allow 443/tcp  >/dev/null 2>&1 || true
    ufw allow 22/tcp   >/dev/null 2>&1 || true
    ok "Firewall configured (ports 22, 80, 443)"
fi

# ── Step 8: Health check ────────────────────────────────────
info "Running health checks..."
sleep 10

HEALTH_URL="http://localhost:8000/health"
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    ok "API health check passed"
else
    warn "API health check failed — the app may still be starting up (model loading takes time)"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Nexus Conversational AI — Deployment Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$SKIP_SSL" == true ]]; then
    echo -e "  🌐  Frontend:   ${BLUE}http://$DOMAIN${NC}"
    echo -e "  📡  API:        ${BLUE}http://$DOMAIN/api/v1/chat${NC}"
    echo -e "  📖  API Docs:   ${BLUE}http://$DOMAIN/docs${NC}"
else
    echo -e "  🌐  Frontend:   ${BLUE}https://$DOMAIN${NC}"
    echo -e "  📡  API:        ${BLUE}https://$DOMAIN/api/v1/chat${NC}"
    echo -e "  📖  API Docs:   ${BLUE}https://$DOMAIN/docs${NC}"
    echo -e "  🔒  SSL:        Let's Encrypt (auto-renewing)"
fi
echo -e "  🩺  Health:     ${BLUE}https://$DOMAIN/health${NC}"
echo ""
echo -e "  Useful commands:"
echo -e "    docker compose logs -f          # View logs"
echo -e "    docker compose ps               # Check status"
echo -e "    docker compose restart           # Restart services"
echo -e "    docker compose down              # Stop everything"
echo ""
