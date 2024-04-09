
from .model_loss import logits_to_tokens_loss, loss_fn
from .model_token_to_char import tokens_to_string

from .useful_node import NodeLocation
from .useful_node import position_name, location_name, answer_name, NodeLocation, UsefulNode, UsefulNodeList

from .quanta_constants import QType, NO_IMPACT_TAG
from .quanta_map_impact import get_question_answer_impact, sort_unique_digits
from .quanta_filter import FilterNode, FilterAnd, FilterOr, FilterHead, FilterNeuron, FilterContains, FilterPosition, FilterAttention, FilterImpact, FilterPCA, FilterAlgo, filter_nodes

from .ablate_config import AblateConfig, acfg
from .ablate_hooks import a_predict_questions, a_run_attention_intervention

from .maths_constants import MathsToken, MathsBehavior, MathsAlgorithm 
from .maths_data_generator import maths_data_generator, maths_data_generator_core, make_maths_questions_and_answers
from .maths_complexity import get_maths_question_complexity
from .maths_utilities import int_to_answer_str


def run_intervention_core(cfg, node_locations, store_question, clean_question, operation, expected_answer_impact, expected_answer_int, strong):
    assert(len(node_locations) > 0)
    assert(store_question[0] < + 10 ** cfg.n_digits)
    assert(store_question[1] > - 10 ** cfg.n_digits)
    assert(store_question[0] < + 10 ** cfg.n_digits)
    assert(store_question[1] > - 10 ** cfg.n_digits)
    assert(clean_question[0] < + 10 ** cfg.n_digits)
    assert(clean_question[1] > - 10 ** cfg.n_digits)
    assert(clean_question[0] < + 10 ** cfg.n_digits)
    assert(clean_question[1] > - 10 ** cfg.n_digits)

    # Calculate the test (clean) question answer e.g. "+006671"
    clean_answer_int = clean_question[0]+clean_question[1] if operation == MathsToken.PLUS else clean_question[0]-clean_question[1]
    clean_answer_str = int_to_answer_str(cfg, clean_answer_int)
    expected_answer_str = int_to_answer_str(cfg, expected_answer_int)

    # Matrices of tokens
    store_question_and_answer = make_maths_questions_and_answers(cfg, acfg.operation, QType.UNKNOWN, MathsBehavior.UNKNOWN, [store_question])
    clean_question_and_answer = make_maths_questions_and_answers(cfg, acfg.operation, QType.UNKNOWN, MathsBehavior.UNKNOWN, [clean_question])

    acfg.reset_intervention(expected_answer_str, expected_answer_impact, operation)
    acfg.ablate_node_locations = node_locations

    run_description = a_run_attention_intervention(cfg, store_question_and_answer, clean_question_and_answer, clean_answer_str)

    acfg.ablate_description = "Intervening on " + acfg.node_names() + ", " + ("Strong" if strong else "Weak") + ", Node[0]=" + acfg.ablate_node_locations[0].name() + ", " + run_description


# Run an intervention where we have a precise expectation of the intervention impact
def run_strong_intervention(cfg, node_locations, store_question, clean_question, operation, expected_answer_impact, expected_answer_int):

    # These are the actual model prediction outputs (while applying our node-level intervention).
    run_intervention_core(cfg, node_locations, store_question, clean_question, operation, expected_answer_impact, expected_answer_int, True)

    answer_success = (acfg.intervened_answer == acfg.expected_answer)
    impact_success = (acfg.intervened_impact == acfg.expected_impact)
    success = answer_success and impact_success

    if acfg.show_test_failures and not success:
        print("Failed: " + acfg.ablate_description)
    if acfg.show_test_successes and success:
        print("Success: " + acfg.ablate_description)

    return success, answer_success, impact_success


# Run an intervention where we expect the intervention to have a non-zero impact but we cant precisely predict the answer impact
def run_weak_intervention(cfg, node_locations, store_question, clean_question, operation):

    # Calculate the test (clean) question answer e.g. "+006671"
    expected_answer_int = clean_question[0]+clean_question[1] if operation == MathsToken.PLUS else clean_question[0]-clean_question[1]

    run_intervention_core(cfg, node_locations, store_question, clean_question, operation, NO_IMPACT_TAG, expected_answer_int, False)

    success = not ((acfg.intervened_answer == acfg.expected_answer) or (acfg.intervened_impact == NO_IMPACT_TAG))

    if acfg.show_test_failures and not success:
        print("Failed: Intervention had no impact on the answer", acfg.ablate_description)
    if acfg.show_test_successes and success:
        print("Success: " + acfg.ablate_description)

    return success


# A test function that always suceeds 
def succeed_test(node_locations, alter_digit, strong):
    print( "Test confirmed", node_locations[0].name(), node_locations[1].name() if len(node_locations)>1 else "", "" if strong else "Weak")
    return True


# Common set of node filters (pre-requisites) for some maths tasks based on token position, attention to Dn and D'n, and answer digit impact
def math_common_prereqs(cfg, position, attend_digit, impact_digit):
    return FilterAnd(
        FilterHead(), # Is an attention head
        FilterPosition(position_name(position)), # Is at token position Px
        FilterAttention(cfg.dn_to_position_name(attend_digit)), # Attends to Dn
        FilterAttention(cfg.ddn_to_position_name(attend_digit)), # Attends to D'n
        FilterImpact(answer_name(impact_digit))) # Impacts Am


def add_ss_tag(impact_digit):
    return answer_name(impact_digit-1)  + "." + MathsAlgorithm.ADD_S_TAG.value


# These rules are prerequisites for (not proof of) an Addition UseSum9 node
def add_ss_prereqs(position, impact_digit):
    # Impacts An and pays attention to Dn-2 and D'n-2
    return math_common_prereqs( position, impact_digit-2, impact_digit)

