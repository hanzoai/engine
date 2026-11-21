use hanzo_worlds::{GenerativeModel, Request, WorldInference};
fn world() -> WorldInference {
    let id = vec![vec![1., 0.], vec![0., 1.]];
    let m = GenerativeModel::new(
        vec![id.clone(), id.clone()],
        vec![id, vec![vec![0., 1.], vec![1., 0.]]],
        vec![0.9, 0.1],
    )
    .unwrap();
    WorldInference::new(m, 4.)
}
#[test]
fn infer_observe_then_plan_without_execution() {
    let r = world()
        .infer(
            &Request {
                belief: vec![0.5, 0.5],
                observation: Some((0, 0)),
                policies: vec![vec![0], vec![1]],
            },
            &[0.5, 0.5],
            &[true, true],
        )
        .unwrap();
    assert_eq!(r.posterior.unwrap().belief, vec![1., 0.]);
    assert_eq!(r.policies.selected, 0);
    assert!(!r.execution_authorized);
}
#[test]
fn host_mask_cannot_be_overridden_by_request() {
    let req = Request {
        belief: vec![1., 0.],
        observation: None,
        policies: vec![vec![0], vec![1]],
    };
    assert_eq!(
        world()
            .infer(&req, &[0.5, 0.5], &[false, true])
            .unwrap()
            .policies
            .selected,
        1
    );
    assert!(world().infer(&req, &[0.5, 0.5], &[false, false]).is_err());
    let malicious = r#"{"belief":[1,0],"policies":[[0]],"observation":null,"admissible":[true]}"#;
    assert!(serde_json::from_str::<Request>(malicious).is_err());
}
