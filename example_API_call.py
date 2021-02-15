'''
FLASK_ENV=development FLASK_APP=screener_app.py flask run
'''

import requests
import json
import time

data = [
  {'ti': 'Reactive stroma in prostate cancer progression.',
   'abs': 'The development of an altered stromal microenvironment in response to carcinoma is a common feature of many tumors. We reviewed the literature describing characteristics of reactive stroma, how reactive stroma affects cancer progression and how carcinoma regulates reactive stroma. Moreover, we present a hypothesis of reactive stroma in prostate cancer and discuss how the biology of reactive stroma may be used in novel diagnostic and therapeutic approaches.\n                An extensive literature search was performed to review reports of the general features of wound repair stroma, general stromal responses to carcinoma, and stromal biology of normal and prostate cancer tissues. These studies were analyzed and a reactive stroma hypothesis in prostate cancer was developed.\n                Modifications to the stroma of breast, colon and prostate tumors parallel the generation of granulation tissue in wound repair. These changes include stromal cell phenotypic switching, extracellular matrix remodeling and angiogenesis induction. Therefore, it is predicted that a modified wound healing response induces the formation of reactive stroma in cancer to create a tumor promoting environment. Based on its role in wound repair and its over expression in prostate cancer, transforming growth factor-beta stands out as a potential regulator of reactive stroma.\n                Reactive stroma in prostate cancer and granulation tissue in wound repair show similar biological responses and processes that are predicted to promote cancer progression. Further identification of specific functional and regulatory mechanisms in prostate cancer reactive stroma may aid in the use of reactive stroma for novel diagnostic and therapeutic approaches.',
   'label': '1',
  },
  {'ti': 'Does usage of a parachute in contrast to free fall prevent major trauma?: a prospective randomised-controlled trial in rag dolls.',
   'abs': 'PURPOSE: It is undisputed for more than 200 years that the use of a parachute prevents major trauma when falling from a great height. Nevertheless up to date no prospective randomised controlled trial has proven the superiority in preventing trauma when falling from a great height instead of a free fall. The aim of this prospective randomised controlled trial was to prove the effectiveness of a parachute when falling from great height. METHODS: In this prospective randomised-controlled trial a commercially acquirable rag doll was prepared for the purposes of the study design as in accordance to the Declaration of Helsinki, the participation of human beings in this trial was impossible. Twenty-five falls were performed with a parachute compatible to the height and weight of the doll. In the control group, another 25 falls were realised without a parachute. The main outcome measures were the rate of head injury; cervical, thoracic, lumbar, and pelvic fractures; and pneumothoraxes, hepatic, spleen, and bladder injuries in the control and parachute groups. An interdisciplinary team consisting of a specialised trauma surgeon, two neurosurgeons, and a coroner examined the rag doll for injuries. Additionally, whole-body computed tomography scans were performed to identify the injuries. RESULTS: All 50 falls-25 with the use of a parachute, 25 without a parachute-were successfully performed. Head injuries (right hemisphere p = 0.008, left hemisphere p = 0.004), cervical trauma (p < 0.001), thoracic trauma (p < 0.001), lumbar trauma (p < 0.001), pelvic trauma (p < 0.001), and hepatic, spleen, and bladder injures (p < 0.001) occurred more often in the control group. Only the pneumothoraxes showed no statistically significant difference between the control and parachute groups. CONCLUSIONS: A parachute is an effective tool to prevent major trauma when falling from a great height.',
   'label': '0',
  },
]


base_url="http://127.0.0.1:5000/"
headers = {'Content-Type': 'application/json', 'Accept':'application/json'}

## for training
requests.post(base_url+'train/1338', json=json.dumps({"labeled_data":data}), headers=headers)

### for testing
predictions = requests.post(base_url+'predict/vaccine_model', json=json.dumps({"input_citations":data}), headers=headers)
