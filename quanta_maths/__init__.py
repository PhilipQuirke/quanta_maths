from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_model_loader import (build_maths_config, load_maths_model_from_hf,
    make_untrained_control, DEFAULT_HF_REPO)
from quanta_maths.maths_constants import MathsBehavior, MathsToken, MathsTask, maths_tokens_to_names, maths_tokens_to_names
from quanta_maths.maths_utilities import set_maths_vocabulary, set_maths_question_meanings, int_to_answer_str, tokens_to_unsigned_int, tokens_to_answer
from quanta_maths.maths_data_generator import maths_data_generator_addition, maths_data_generator_subtraction, maths_data_generator_multiplication, maths_data_generator, maths_data_generator_mixed, make_maths_questions_and_answers, MixedMathsDataset, get_mixed_maths_dataloader
from quanta_maths.maths_search_add import add_ss_functions, add_sc_functions, add_sa_functions, add_st_functions, add_stc_functions
from quanta_maths.maths_search_sub import sub_mt_functions, sub_gt_functions, sub_mb_functions, sub_md_functions, neg_nd_functions, neg_nb_functions
from quanta_maths.maths_search_mix import run_strong_intervention, run_weak_intervention, SubTaskBaseMath, opr_functions, sgn_functions
from quanta_maths.maths_pca import _build_title_and_error_message, pca_op_tag, plot_pca_for_an, manual_nodes_pca, plot_nodes_pca_start_core, plot_nodes_pca_end

from quanta_maths.MathsTestQuestions.tricase_test_questions_generator import (
    TOTAL_TRICASE_QUESTIONS, make_maths_tricase_questions, make_maths_tricase_questions_customized)
from quanta_maths.MathsTestQuestions.manual_test_questions_generator import make_maths_test_questions_and_answers
from quanta_maths.MathsTestQuestions.test_questions_checker import (test_maths_questions_by_complexity, test_maths_questions_by_impact, 
    test_maths_questions_and_add_useful_node_tags, test_correctness_on_num_questions, test_correctness_on_num_questions_core)

from quanta_maths.maths_complexity import (SimpleQuestionDescriptor, get_maths_min_complexity, get_maths_question_complexity, 
    calc_maths_quanta_for_position_nodes, get_maths_node_operation_coverage, get_maths_nodes_operation_coverage, get_maths_operation_complexity)

from quanta_maths.model_sae_train import analyze_mlp_with_sae, optimize_sae_hyperparameters
from quanta_maths.model_sae_graph import analyze_and_visualize_sae

from quanta_maths.maths_probe import (sub_labels, TASK_CHANCE, ALL_SITES, site_hook_and_pos,
    collect_site_activations, train_test_split_idx, balance_idx, fit_probe,
    probe_balanced_accuracy, probe_accuracy_with_null, class_mean_subspace, principal_angles_deg,
    real_dft_basis, marginal_dft_spectrum, freq1_plane_share, unique_linear_share,
    pc_plane_coords, angular_order_stat, wraparound_ratio, participation_ratio,
    dft_permutation_pvalues)

from quanta_maths.maths_edge_patch import (answer_positions, consuming_pos, ln_scale, head_ov,
    head_edge_delta, direct_path_delta, run_edge_patch, run_multi_head_edge_patch,
    pattern_patch_prediction, synthetic_redirect_prediction)

from quanta_maths.maths_batch import (ACCURATE_MODELS, run_batch, tag_stc_nodes, tag_linxfer_nodes)