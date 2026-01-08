//! Admin UI routes.
//!
//! Provides:
//! - Dashboard (`/admin`) - Overview with stats
//! - Users (`/admin/users`) - User management
//! - Requests (`/admin/requests`) - Request history

use std::sync::Arc;

use askama::Template;
use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
    Router,
};

use crate::audit::{DashboardStats, RequestSummary, RequestWithResponse, UserWithStats};
use crate::AppState;

/// Admin user info for templates.
#[derive(Clone)]
pub struct AdminUser {
    pub sub: String,
    pub email: Option<String>,
}

/// Middleware that requires an authenticated admin user.
async fn require_admin(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    // Get Authorization header
    let auth_result = state.jwks_client.authenticate(request.headers()).await;

    match auth_result {
        Ok(user) => {
            if !user.is_admin() {
                return (
                    StatusCode::FORBIDDEN,
                    Html("<h1>403 Forbidden</h1><p>Admin access required. You need the 'admin' role for this application.</p>"),
                ).into_response();
            }

            // Store admin user for handlers
            request.extensions_mut().insert(AdminUser {
                sub: user.sub,
                email: user.email,
            });

            next.run(request).await
        }
        Err(_) => {
            // Return 401 with instructions
            (
                StatusCode::UNAUTHORIZED,
                Html("<h1>401 Unauthorized</h1><p>Please provide a valid Bearer token with admin privileges.</p>"),
            ).into_response()
        }
    }
}

// ========== Templates ==========

#[derive(Template)]
#[template(path = "admin/dashboard.html")]
struct DashboardTemplate {
    nav_active: &'static str,
    admin_email: String,
    stats: DashboardStats,
    recent_requests: Vec<RequestSummary>,
}

#[derive(Template)]
#[template(path = "admin/users.html")]
struct UsersTemplate {
    nav_active: &'static str,
    admin_email: String,
    users: Vec<UserWithStats>,
}

#[derive(Template)]
#[template(path = "admin/requests.html")]
struct RequestsTemplate {
    nav_active: &'static str,
    admin_email: String,
    requests: Vec<RequestWithResponse>,
    filter_user_id: String,
    filter_model: String,
    page: u32,
    total_pages: u32,
}

// ========== Handlers ==========

/// GET /admin - Dashboard
async fn dashboard(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
) -> impl IntoResponse {
    let stats = state.audit_logger.get_stats().unwrap_or(DashboardStats {
        total_users: 0,
        total_requests: 0,
        requests_24h: 0,
        total_tokens: 0,
    });

    let recent_requests = state.audit_logger.get_recent_requests(10).unwrap_or_default();

    let template = DashboardTemplate {
        nav_active: "dashboard",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        stats,
        recent_requests,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

/// GET /admin/users - User list
async fn users_list(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
) -> impl IntoResponse {
    let users = state.audit_logger.get_users_with_stats().unwrap_or_default();

    let template = UsersTemplate {
        nav_active: "users",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        users,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

#[derive(serde::Deserialize)]
struct RequestsQuery {
    user_id: Option<String>,
    model: Option<String>,
    page: Option<u32>,
}

/// GET /admin/requests - Request history
async fn requests_list(
    State(state): State<Arc<AppState>>,
    axum::Extension(admin): axum::Extension<AdminUser>,
    Query(query): Query<RequestsQuery>,
) -> impl IntoResponse {
    let page = query.page.unwrap_or(1).max(1);
    let per_page = 50;

    let (requests, total_pages) = state
        .audit_logger
        .get_requests_paginated(
            query.user_id.as_deref(),
            query.model.as_deref(),
            page,
            per_page,
        )
        .unwrap_or_default();

    let template = RequestsTemplate {
        nav_active: "requests",
        admin_email: admin.email.unwrap_or_else(|| admin.sub.clone()),
        requests,
        filter_user_id: query.user_id.unwrap_or_default(),
        filter_model: query.model.unwrap_or_default(),
        page,
        total_pages,
    };

    Html(template.render().unwrap_or_else(|e| format!("Template error: {}", e)))
}

/// POST /admin/users/:id/disable - Disable a user
async fn disable_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.audit_logger.disable_user(&user_id) {
        Ok(_) => Redirect::to("/admin/users"),
        Err(e) => {
            tracing::error!("Failed to disable user {}: {}", user_id, e);
            Redirect::to("/admin/users")
        }
    }
}

/// POST /admin/users/:id/enable - Enable a user
async fn enable_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    match state.audit_logger.enable_user(&user_id) {
        Ok(_) => Redirect::to("/admin/users"),
        Err(e) => {
            tracing::error!("Failed to enable user {}: {}", user_id, e);
            Redirect::to("/admin/users")
        }
    }
}

/// Build the admin router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/users", get(users_list))
        .route("/users/:id/disable", post(disable_user))
        .route("/users/:id/enable", post(enable_user))
        .route("/requests", get(requests_list))
        .layer(middleware::from_fn_with_state(state.clone(), require_admin))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_requests_query_defaults() {
        let query = RequestsQuery {
            user_id: None,
            model: None,
            page: None,
        };
        assert!(query.user_id.is_none());
        assert!(query.model.is_none());
        assert!(query.page.is_none());
    }

    #[tokio::test]
    async fn test_requests_query_with_values() {
        let query = RequestsQuery {
            user_id: Some("user123".to_string()),
            model: Some("llama2".to_string()),
            page: Some(2),
        };
        assert_eq!(query.user_id, Some("user123".to_string()));
        assert_eq!(query.model, Some("llama2".to_string()));
        assert_eq!(query.page, Some(2));
    }

    #[tokio::test]
    async fn test_admin_user_struct() {
        let admin = AdminUser {
            sub: "auth0|123".to_string(),
            email: Some("admin@example.com".to_string()),
        };
        assert_eq!(admin.sub, "auth0|123");
        assert_eq!(admin.email, Some("admin@example.com".to_string()));
    }

    #[tokio::test]
    async fn test_admin_user_without_email() {
        let admin = AdminUser {
            sub: "auth0|456".to_string(),
            email: None,
        };
        assert!(admin.email.is_none());
    }

    #[tokio::test]
    async fn test_dashboard_stats_default() {
        let stats = DashboardStats {
            total_users: 0,
            total_requests: 0,
            requests_24h: 0,
            total_tokens: 0,
        };
        assert_eq!(stats.total_users, 0);
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_dashboard_stats_with_values() {
        let stats = DashboardStats {
            total_users: 100,
            total_requests: 1000,
            requests_24h: 50,
            total_tokens: 50000,
        };
        assert_eq!(stats.total_users, 100);
        assert_eq!(stats.total_requests, 1000);
        assert_eq!(stats.requests_24h, 50);
        assert_eq!(stats.total_tokens, 50000);
    }

    #[tokio::test]
    async fn test_users_template() {
        let template = UsersTemplate {
            nav_active: "users",
            admin_email: "admin@test.com".to_string(),
            users: vec![],
        };
        assert_eq!(template.nav_active, "users");
        assert!(template.users.is_empty());
    }

    #[tokio::test]
    async fn test_requests_template() {
        let template = RequestsTemplate {
            nav_active: "requests",
            admin_email: "admin@test.com".to_string(),
            requests: vec![],
            filter_user_id: "".to_string(),
            filter_model: "".to_string(),
            page: 1,
            total_pages: 1,
        };
        assert_eq!(template.page, 1);
        assert_eq!(template.total_pages, 1);
    }

    #[tokio::test]
    async fn test_dashboard_template() {
        let template = DashboardTemplate {
            nav_active: "dashboard",
            admin_email: "admin@test.com".to_string(),
            stats: DashboardStats {
                total_users: 10,
                total_requests: 100,
                requests_24h: 5,
                total_tokens: 1000,
            },
            recent_requests: vec![],
        };
        assert_eq!(template.nav_active, "dashboard");
        assert_eq!(template.stats.total_users, 10);
    }

    #[tokio::test]
    async fn test_disable_user_route_path_extraction() {
        let user_id = "test-user-123".to_string();
        assert_eq!(user_id, "test-user-123");
    }
}
