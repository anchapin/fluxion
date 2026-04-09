pub struct GitHubClient {
    token: Option<String>,
}

impl GitHubClient {
    pub fn new(token: Option<String>) -> Self {
        Self { token }
    }
}
