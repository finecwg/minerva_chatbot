__author__ = 'finecwg'

from chatting import MinervaQA
minerva_qa = MinervaQA()

query = "Can I bring alcohol to the res hall?"

response = minerva_qa.answer_query(query)