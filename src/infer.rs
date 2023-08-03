use llm::Model;

#[derive(Clone)]
struct InferenceState<'a> {
    // session: InferenceSession,
    model: &'a dyn Model,
}
