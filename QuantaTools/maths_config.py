import re

from .useful_config import UsefulConfig
from .useful_node import position_name, answer_name


# Extends UsefulConfig with mathematics-specific info for "123456+123456=+0246912" style questions
class MathsConfig(UsefulConfig):

    def __init__(self):
        super().__init__()

        # Percent of questions that are multiplication, subtraction (rest are addition questions).
        self.perc_mult = 0 # e.g. 20
        self.perc_sub = 0 # e.g. 80

        # Save graphs to CoLab temp files as PDF or SVG. You can manually export temp files for re-use in papers.
        self.graph_file_suffix = "svg"
        
        self.n_digits = 6
        self.initialize_token_positions( self.question_tokens(), self.answer_tokens(), self.answer_meanings_ascend )


    def perc_add(self):
        return max(0, 100 - self.perc_mult - self.perc_sub)


    # The number of question tokens
    # This is also the token position of the first answer digit (which is a "+" or a  "-")
    def question_tokens(self):
        return self.n_digits*2 + 2

    def answer_tokens(self):
        return self.n_digits + 2

    def n_ctx(self):
        return self.question_tokens() + self.answer_tokens()


    # How many slices do we break the MLP layer up into?
    def mlp_slices(self):
        return 1 # Paper 2 used this granualarity
        # return self.n_heads * self.d_mlp_multiplier # Alternative for Paper 3?
  

    # Maths question and answer token position meanings are D5, .., D0, *, D5', .., D0', =, A7, A6, .., A0      
    # Convert D0 to P5, D1 to P4, D2 to P3 in 6 digit addition
    def dn_to_position_name(self, n):
        return position_name(self.n_digits - 1 - n) 
    # Convert D'0 to P10, D'1 to P9, D'2 to P8, etc in 6 digit addition
    def ddn_to_position_name(self, n):
      return position_name(2 * self.n_digits - n) 
    # Convert A0 to P20, A1 to P19, A2 to P18, etc in 6 digit addition
    def an_to_position_name(self, n):
      return position_name(self.n_ctx() - 1 - n)
    # Position of the operator (+, -, * or /)
    def op_position_name(self):
      return position_name(self.n_digits)


    def parse_model_name(self):
        super().parse_model_name()
        
        match = re.search("d(\d)_", self.model_name)
        if match:
            self.n_digits = int(match.group(1))
            
        self.initialize_token_positions( self.question_tokens(), self.answer_tokens(), self.answer_meanings_ascend )  



    def short_config_description(self):       
        return f'_d{self.n_digits}' + super().short_config_description()      
    

    def op_config_description(self):
        return 'mul' if self.perc_mult == 100 else 'sub' if self.perc_sub == 100 else 'add' if self.perc_add() == 100 else 'mix'    
    

    def file_config_prefix(self):
        return self.insert_config_description() + self.op_config_description() + self.long_config_description()
