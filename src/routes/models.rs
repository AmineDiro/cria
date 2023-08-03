use crate::ModelList;
use axum::{extract::State, Json};

pub async fn get_models(State(state): State<ModelList>) -> Json<ModelList> {
    Json(state)
}
