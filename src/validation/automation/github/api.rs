pub struct GitHubClient {
    #[allow(dead_code)]
    token: Option<String>,
}

impl GitHubClient {
    pub fn new(token: Option<String>) -> Self {
        Self { token }
    }
}
