from QuantaMechInterp import (QType, a_run_attention_intervention, NO_IMPACT_TAG, SubTaskBase, position_name, answer_name,
    FilterAnd, FilterHead, FilterPosition, FilterAttention, FilterImpact, FilterContains, QCondition)
from MathsMechInterp.maths_constants import MathsToken, MathsBehavior, MathsTask 
from MathsMechInterp.maths_search_mix import run_strong_intervention, run_weak_intervention, SubTaskBaseMath
from MathsMechInterp.maths_utilities import digit_name


# Addition "Use Sum 9" (SS) sub-task e.g. 34633+55555=+090188 where D4 and D'4 sum to 9 (4+5), and D3 + D'3 > 10
# Node output is likely binary (aka boolean)    
class add_ss_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.PLUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit-1)  + "." + MathsTask.SS_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn-2 and D'n-2. Impacts An
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit-2, impact_digit)

    @staticmethod
    def test1(cfg, alter_digit):
        # 25222 + 44444 = 69666. Has no Dn-2.SC but has Dn-1.SS so not a UseSum9 case
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(4)]
        store_question[0] += (5-2) * 10 ** (alter_digit - 1)

        # 34633 + 55555 = 90188. Has Dn-2.SC and Dn-1.SS so is a UseSum9 case
        clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(5)]
        clean_question[0] += (4-3) * 10 ** (alter_digit - 1)
        clean_question[0] += (6-3) * 10 ** (alter_digit - 2)

        # When we intervene we expect answer 80188
        intervened_answer = clean_question[0] + clean_question[1] - 10 ** (alter_digit)

        return store_question, clean_question, intervened_answer

    @staticmethod
    # Intervention ablation test for addition "Use Sum 9" (SS) task
    def test(cfg, acfg, alter_digit, strong):
        if alter_digit < 2 or alter_digit > cfg.n_digits:
            acfg.reset_intervention()
            return False

        store_question, clean_question, intervened_answer = add_ss_functions.test1(cfg, alter_digit)

        intervention_impact = answer_name(alter_digit)
    
        success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        if success:
            print( "Test confirmed", acfg.ablate_node_names, "perform", add_ss_functions.tag(alter_digit), "impacting", intervention_impact, "accuracy.", "" if strong else "Weak")

        return success


# Addition "Make Carry 1" (SC) sub-task e.g. 222222+666966=+0889188 where D2 + D'2 > 10
# Node output is binary (aka boolean)    
class add_sc_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.PLUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit-1)  + "." + MathsTask.SC_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn-1 and D'n-1. Impacts An
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit-1, impact_digit)
            
    @staticmethod
    def test(cfg, acfg, impact_digit, strong):
        alter_digit = impact_digit - 1

        if alter_digit < 0 or alter_digit >= cfg.n_digits:
            acfg.reset_intervention()
            return False

        intervention_impact = answer_name(impact_digit)

        # 222222 + 666966 = 889188. Has Dn.SC
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(6)]
        store_question[1] += (9 - 6) * (10 ** alter_digit)

        # 333333 + 555555 = 888888. No Dn.SC
        clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(5)]

        # When we intervene we expect answer 889888
        intervened_answer = clean_question[0] + clean_question[1] + 10 ** (alter_digit+1)

        success, _, _ = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        if success:
            print( "Test confirmed", acfg.ablate_node_names, "perform", add_sc_functions.tag(alter_digit), "impacting", intervention_impact, "accuracy.", "" if strong else "Weak")

        return success


# Addition "Simple Add" (SA) sub-task e.g. 555555+111111=+0666666 where D3 + D'3 < 10
# Node output is one of 10 values 0 to 9
class add_sa_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.PLUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit) + "." + MathsTask.SA_TAG.value

    @staticmethod
    def prereqs(cfg, position, impact_digit):
        # Pays attention to Dn and D'n. Impacts An
        return SubTaskBaseMath.math_latetoken_subtask_prereqs(cfg, position, impact_digit, impact_digit)

    @staticmethod
    def test1(cfg, alter_digit):
        # 222222 + 111111 = +333333. No Dn.SC
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(1)]

        # 555555 + 444444 = +999999. No Dn.SC
        clean_question = [cfg.repeat_digit(5), cfg.repeat_digit(4)]

        # When we intervene we expect answer +999399
        intervened_answer = clean_question[0] + clean_question[1] + (3-9) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test2(cfg, alter_digit):
        # 222222 + 666666 = +888888. No Dn.SC
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(6)]

        # 555555 + 111111 = +666666. No Dn.SC
        clean_question = [cfg.repeat_digit(5), cfg.repeat_digit(1)]

        # When we intervene we expect answer +666866
        intervened_answer = clean_question[0] + clean_question[1] + (8-6) * 10 ** alter_digit

        return store_question, clean_question, intervened_answer

    @staticmethod
    def test(cfg, acfg, alter_digit, strong):
        # Note: MD and SA give the same result when D'=0 or D=D'=5. We avoid ablation tests like this.

        intervention_impact = answer_name(alter_digit)

        store_question, clean_question, intervened_answer = add_sa_functions.test1(cfg, alter_digit)
        success1, _, impact_success1 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        store_question, clean_question, intervened_answer = add_sa_functions.test2(cfg, alter_digit)
        success2, _, impact_success2 = run_strong_intervention(cfg, acfg, store_question, clean_question, intervention_impact, intervened_answer)

        success = (success1 and success2) if strong else (impact_success1 and impact_success2)

        if success:
            print( "Test confirmed:", acfg.ablate_node_names, "perform", add_sa_functions.tag(alter_digit), "= (D"+str(alter_digit)+" + D'"+str(alter_digit)+") % 10 impacting "+intervention_impact+" accuracy.", "" if strong else "Weak", acfg.intervened_answer)

        return success


# Addition "Essential Carry Info" (ST) sub-task. 
# Found in early tokens. Node output is tricase: ST8, ST9, ST10    
# Has impact "A65432" to "A65" (Always excludes A0. Generally excludes A1 which uses SC).
class add_st_functions(SubTaskBaseMath):

    @staticmethod
    def operation():
        return MathsToken.PLUS
    
    @staticmethod
    def tag(impact_digit):
        return answer_name(impact_digit) + "." + MathsTask.ST_TAG.value

    @staticmethod
    def prereqs(cfg, position, focus_digit):
        # Example meaning: 
        #   And(IsHead, 
        #       Position>=n_digits, Position<=num_question_positions, Position=14,
        #       AttendsTo:D3, AttendsTo:D'3, 
        #       MAY have bi- or trigram PCA for addition questions,
        #       Impacts addition questions)    
        return FilterAnd(
            FilterHead(),
            FilterPosition(position_name(cfg.n_digits), QCondition.MIN), # Occurs from the operator token
            FilterPosition(position_name(cfg.num_question_positions), QCondition.MAX), # Occurs by the = token
            FilterPosition(position_name(position)), # Is at token position Px
            FilterAttention(cfg.dn_to_position_name(focus_digit)), # Attends to Dn
            FilterAttention(cfg.ddn_to_position_name(focus_digit)), # Attends to D'n
            FilterContains(QType.MATH_ADD, MathsBehavior.ADD_PCA_TAG.value, QCondition.MAY), # Weak: Node PCA is interpretable (bigram or trigram output) with respect to addition 1,U,0
            FilterContains(QType.MATH_ADD, MathsBehavior.ADD_COMPLEXITY_PREFIX.value)) # Impacts addition questions

    @staticmethod
    def test(cfg, acfg, focus_digit, strong):
        # 222222 + 777977 = 1000188. Has Dn.SC
        store_question = [cfg.repeat_digit(2), cfg.repeat_digit(7)]
        store_question[1] += (9 - 7) * (10 ** focus_digit)

        # 333333 + 666666 = 999999. No Dn.SC
        clean_question = [cfg.repeat_digit(3), cfg.repeat_digit(6)]

        success = run_weak_intervention(cfg, acfg, store_question, clean_question)

        if success:
            description = acfg.ablate_node_names + " perform " + add_st_functions.tag(focus_digit) + " = TriCase(D"+str(focus_digit)+" + D'"+str(focus_digit)+")"
            print("Test confirmed", description, "Impact:", acfg.intervened_impact, "" if strong else "Weak")

        return success
