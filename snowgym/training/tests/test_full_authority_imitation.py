import json
import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_HOLD, ACTION_MOVE, ACTION_NOOP, ACTION_THROW
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from snowgym_training.options import full_authority_imitation as im


def fake_prediction(action_logits, move_raw, throw_raw, power_raw):
    return {"living": torch.tensor([[True, True, True, False]]), "action_logits": action_logits,
            "move_raw": move_raw, "throw_raw": throw_raw, "power_raw": power_raw}


class FakeModel:
    """Global decode with fixed head outputs, so each loss term can be checked by hand."""

    def __init__(self, prediction):
        self.prediction = prediction

    def __call__(self, observation, *, with_value=True):
        assert with_value is False
        return self.prediction

    def decode_move(self, observation, latent):
        return torch.tanh(latent)


def observation(units=4):
    allies = torch.zeros(1, units, 21)
    allies[..., 0:2] = 1
    return {"allies": allies, "unit_action_mask": torch.ones(1, units, 4, dtype=torch.int8)}


def test_loss_terms_match_hand_computation_and_respect_label_types():
    cfg = im.configuration()
    # Unit 0 MOVE (far, heading 90 degrees off), unit 1 THROW (aim opposite), unit 2 NOOP, unit 3 dead.
    move_raw = torch.zeros(1, 4, 2)
    move_raw[0, 0] = torch.tensor([0., 5.])          # tanh -> (0, ~1) -> straight "up"
    move_raw[0, 1] = torch.tensor([50., 50.])        # a THROW unit's move head must not be scored
    throw_raw = torch.zeros(1, 4, 2)
    throw_raw[0, 1] = torch.tensor([-5., 0.])        # aim to the left
    power_raw = torch.zeros(1, 4)
    logits = torch.zeros(1, 4, 4)
    model = FakeModel(fake_prediction(logits, move_raw, throw_raw, power_raw))
    labels = {"action_type": torch.tensor([[ACTION_MOVE, ACTION_THROW, ACTION_NOOP, ACTION_HOLD]]),
              "target": torch.tensor([[[.5, 0.], [.5, 0.], [0., 0.], [0., 0.]]]),
              "power": torch.tensor([[0., .9, 0., 0.]])}
    losses = im.imitation_loss(model, observation(), labels, cfg)
    assert losses["type"].item() == pytest.approx(math.log(4))  # uniform logits over 3 living units
    assert losses["moveHeading"].item() == pytest.approx(1., abs=1e-3)  # 90 degrees -> 1 - cos = 1
    assert losses["moveEndpoint"].item() == 0 and losses["moveFar"] == 1 and losses["moveNear"] == 0
    assert losses["throwAim"].item() == pytest.approx(2., abs=1e-3)  # opposite ray -> 1 - cos = 2
    assert losses["power"].item() == pytest.approx((0.5 - 0.9) ** 2)
    assert losses["throwUnits"] == 1 and losses["illegalLabels"] == 0


def test_near_moves_use_clipped_endpoint_error_and_absent_terms_are_zero():
    cfg = im.configuration()
    move_raw = torch.zeros(1, 4, 2)
    move_raw[0, 0] = torch.atanh(torch.tensor([0.2, 0.]))  # 10 world units from a teacher target at the origin
    model = FakeModel(fake_prediction(torch.zeros(1, 4, 4), move_raw, torch.zeros(1, 4, 2), torch.zeros(1, 4)))
    labels = {"action_type": torch.tensor([[ACTION_MOVE, ACTION_NOOP, ACTION_NOOP, ACTION_NOOP]]),
              "target": torch.zeros(1, 4, 2), "power": torch.zeros(1, 4)}
    losses = im.imitation_loss(model, observation(), labels, cfg)
    assert losses["moveNear"] == 1 and losses["moveEndpoint"].item() == pytest.approx(cfg["endpointClip"] ** 2, rel=1e-4)
    assert losses["throwAim"].item() == 0 and losses["power"].item() == 0 and losses["throwUnits"] == 0


def test_illegal_teacher_labels_are_counted_and_excluded():
    cfg = im.configuration()
    obs = observation()
    obs["unit_action_mask"][0, 0, ACTION_THROW] = 0
    model = FakeModel(fake_prediction(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), torch.zeros(1, 4)))
    labels = {"action_type": torch.tensor([[ACTION_THROW, ACTION_NOOP, ACTION_NOOP, ACTION_NOOP]]),
              "target": torch.full((1, 4, 2), .5), "power": torch.full((1, 4), .5)}
    losses = im.imitation_loss(model, obs, labels, cfg)
    assert losses["illegalLabels"] == 1 and losses["throwUnits"] == 0


def test_imitation_optimizer_excludes_log_stds_and_critic():
    model = FullAuthorityPolicyV1(destination="global")
    trained = {id(p) for p in im.imitation_parameters(model)}
    names = {n for n, p in model.named_parameters() if id(p) in trained}
    assert not any(n.endswith("log_std") or n.startswith("critic.") for n in names)
    assert {"action_head.weight", "move_head.0.weight", "throw_head.0.weight", "power_head.0.weight"} <= names


def test_configuration_seeds_budget_and_frozen_choices():
    cfg = im.configuration()
    assert cfg["destination"] == "global" and cfg["optionHorizon"] == 200
    assert cfg["optimizerSeeds"] == cfg["trainingRngs"] == [97101, 97102, 97103]
    assert im.round_seeds(cfg, 0)[0] == 650000 and im.round_seeds(cfg, 4) == list(range(654000, 654128))
    assert im.split_seeds(cfg, "A") == list(range(600000, 600100))
    assert im.split_seeds(cfg, "B") == list(range(601000, 601100))
    horizon = cfg["optionHorizon"]
    bound = (cfg["roundEpisodes"] * horizon * (1 + 3 * (cfg["fits"] - 1)) + 3 * 2 * 100 * horizon
             + 3 * 100 * horizon + 2 * 2 * 100 * horizon
             + 3 * (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon)
    assert bound == 823_200 and bound <= cfg["simulatorBudget"] == 900_000


def test_label_error_reports_per_type_recall_and_head_errors():
    cfg = im.configuration()
    torch.manual_seed(0)
    model = FullAuthorityPolicyV1(destination="global")
    obs = {"allies": torch.zeros(3, 10, 21), "enemies": torch.zeros(3, 10, 21), "projectiles": torch.zeros(3, 64, 9),
           "obstacles": torch.zeros(3, 64, 9), "ally_mask": torch.zeros(3, 10, dtype=torch.int8),
           "enemy_mask": torch.zeros(3, 10, dtype=torch.int8), "projectile_mask": torch.zeros(3, 64, dtype=torch.int8),
           "obstacle_mask": torch.zeros(3, 64, dtype=torch.int8), "unit_action_mask": torch.ones(3, 10, 4, dtype=torch.int8),
           "plan_unit_roles": torch.zeros(3, 10, 3, dtype=torch.int8), "plan_groups": torch.zeros(3, 3, 38),
           "plan_group_mask": torch.zeros(3, 3, dtype=torch.int8), "plan_role_state": torch.zeros(3, 3, 20)}
    obs["ally_mask"][:, 0] = 1
    obs["allies"][:, 0, 0:2] = 1
    labels = {"action_type": torch.tensor([ACTION_MOVE, ACTION_THROW, ACTION_NOOP])[:, None].repeat(1, 10),
              "target": torch.full((3, 10, 2), .6), "power": torch.full((3, 10), .8)}
    error = im.label_error(model, {"observation": obs, "labels": labels}, cfg)
    assert error["units"] == 3
    assert set(error["perType"]) == {"noop", "move", "throw", "hold"}
    assert error["perType"]["throw"]["support"] == 1 and error["perType"]["hold"]["recall"] is None
    assert error["moveHeadingErrorDegrees"] is not None and error["throwAimHeadingErrorDegrees"] is not None
    assert 0 <= error["powerMeanAbsoluteError"] <= 1


def evaluations(det_a, det_b, sto_a, *, contact=.9):
    return {str(s): {"A-deterministic": {"successFraction": det_a, "contactFraction": contact},
                     "B-deterministic": {"successFraction": det_b, "contactFraction": contact},
                     "A-stochastic": {"successFraction": sto_a}} for s in im.configuration()["optimizerSeeds"]}


def test_decision_rules_and_flags_follow_the_declaration():
    cfg = im.configuration()
    ceilings = {"A": {"successFraction": .89}, "B": {"successFraction": .9}}
    critic = {str(s): {"gatePassed": True} for s in cfg["optimizerSeeds"]}
    errors = {str(s): {"perType": {"throw": {"recall": .7}}} for s in cfg["optimizerSeeds"]}
    near = im.decision_rules(evaluations(.8, .75, .78), ceilings, critic, errors, cfg)
    assert near["outcome"] == "near-ceiling" and near["flags"]["criticHealthy"] and not near["flags"]["throwCollapse"]
    headroom = im.decision_rules(evaluations(.5, .55, .2), ceilings, critic, errors, cfg)
    assert headroom["outcome"] == "ppo-headroom" and headroom["flags"]["executionModeGap"]
    mixed = im.decision_rules(evaluations(.8, .1, .8), ceilings, critic, errors, cfg)
    assert mixed["outcome"] == "imitation-failure"  # the more conservative split wins
    weak = {"A": {"successFraction": .4}, "B": {"successFraction": .4}}
    overlap = im.decision_rules(evaluations(.22, .22, .22), weak, critic, errors, cfg)
    assert overlap["splits"]["A"]["outcome"] == "imitation-failure"  # within a split too (amendment A3)
    no_contact = im.decision_rules(evaluations(.8, .8, .8, contact=.2), ceilings, critic, errors, cfg)
    assert no_contact["outcome"] == "imitation-failure"
    collapsed = im.decision_rules(evaluations(.5, .5, .5), ceilings, critic,
                                  {str(s): {"perType": {"throw": {"recall": .2}}} for s in cfg["optimizerSeeds"]}, cfg)
    assert collapsed["flags"]["throwCollapse"]


def test_paired_difference_is_the_mean_with_a_bracketing_interval():
    result = im.paired_difference([1, 1, 0, 1], [1, 0, 0, 0], samples=500, seed=1)
    assert result["mean"] == pytest.approx(.5)
    assert result["interval95"][0] <= .5 <= result["interval95"][1]


def tiny(**overrides):
    return {**im.configuration(), "optionHorizon": 8, "roundEpisodes": 2, "roundBlockWorlds": 2, "fits": 2,
            "stepsPerFit": 3, "imitationMinibatch": 8, "historyEvery": 1, "developmentSplits": {"A": [929000, 929001],
            "B": [929100, 929101]}, "evaluationBlockWorlds": 2, "roundSeedBase": 929200, "optimizerSeeds": [41],
            "trainingRngs": [41], "warmStartTrainEpisodes": 2, "warmStartHeldOutEpisodes": 2, "warmStartEpochs": 1,
            "minibatchSize": 8, "blockWorlds": 2, "trainSeedBase": 929400, "heldOutSeedBase": 929500,
            "bootstrapSamples": 20, "simulatorBudget": 5000, **overrides}


def test_labeled_collection_is_reproducible_under_a_fixed_checkpoint():
    torch.set_num_threads(1)
    cfg = tiny()
    torch.manual_seed(5)
    model = FullAuthorityPolicyV1(destination="global")
    digests = []
    with SnowGymBatchClient() as client:
        for _ in range(2):
            episodes, part = im.collect(client, [929300, 929301], cfg, model=model, source="repro", block_worlds=2,
                                        account=lambda _count: None)
            digests.append((semantic_state_digest(part["observation"]), semantic_state_digest(part["labels"]),
                            [e["finalDecision"] for e in episodes]))
    assert digests[0] == digests[1]
    assert len(part["labels"]["action_type"]) == sum(e["finalDecision"] for e in episodes)


def test_tiny_end_to_end_run_retains_every_declared_artifact(tmp_path):
    cfg = tiny()
    report = im.execute(tmp_path / "run", cfg)
    root = tmp_path / "run"
    seed = root / "seed-41"
    for name in ("fit-0.pt", "fit-1.pt", "fit-history.json", "dataset.json", "label-error.json",
                 "critic-warm-start.json", "critic-warm-start-arrays.npz", "critic-policy.pt"):
        assert (seed / name).exists(), name
    for name in ("round-1", "eval-A-deterministic", "eval-B-deterministic", "eval-A-stochastic", "critic-episodes"):
        assert (seed / name / "episodes.jsonl").exists(), name
    dataset = json.loads((seed / "dataset.json").read_text())
    assert [r["round"] for r in dataset["rounds"]] == [0, 1]
    assert dataset["aggregateRows"] == sum(r["rows"] for r in dataset["rounds"])
    checkpoint = torch.load(seed / "fit-1.pt")
    assert {"model", "optimizer", "fit", "rows"} <= set(checkpoint)
    assert report["decisionRules"]["outcome"] in {"near-ceiling", "ppo-headroom", "imitation-failure"}
    assert report["simulatorDecisions"] <= cfg["simulatorBudget"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert "report.json" in manifest["artifacts"] and "controls/summary.json" in manifest["artifacts"]
    with pytest.raises(FileExistsError):
        im.execute(root, cfg)


def test_budget_guard_stops_an_over_budget_run(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        im.execute(tmp_path / "capped", tiny(simulatorBudget=10))
