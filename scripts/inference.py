from transformers import pipeline
import json
from VUDENC import utils

#F IXME checkpoint hardgecoded und final_checkpoint missing
classifier_cp_1 = pipeline(task="text-classification", model="saved_models/summarize_python" + "/checkpoint-860") 
#classifier_f_cp = pipeline(task="text-classification", model=args.save_dir + "/final_checkpoint") # Dateien fehlen

test_text = "tokenized_datasets = tokenized_datasets.remove_columns(['snippet_id']) tokenized_datasets = tokenized_datasets.rename_column('label', 'labels') tokenized_datasets.set_format('torch')"
vul_snippet = "SQL_RECURSIVE_QUERY_EDUCATION_GROUP='''\\ WITH RECURSIVE group_element_year_parent AS( SELECT id, child_branch_id, child_leaf_id, parent_id, 0 AS level FROM base_groupelementyear WHERE parent_id IN({list_root_ids'"
not_vul_snippet = "' INNER JOIN group_element_year_parent AS parent on parent.child_branch_id=child.parent_id ) SELECT * FROM group_element_year_parent ; ''''''''' class GroupElementYearManager(models.Manager): def get_queryset"


print("test_text_cp_1: ", classifier_cp_1(test_text))
#print("test text_f_cp: ", classifier_f_cp(test_text))
print("vul_cp_1: ", classifier_cp_1(vul_snippet))
print("not_vul_cp 1: ", classifier_cp_1(not_vul_snippet))


print("########### Test with Github code ##############")

mode = 'sql'
nr = '1'

rep = ""
com = ""
myfile = ""

with open('../VUDENC_data/plain_' + mode, 'r') as infile:  # This date is used for demonstration purpose only
  data = json.load(infile)                      

identifying = utils.getIdentifiers(mode, nr)  
info = utils.getFromDataset(identifying, data)          
sourcefull = info[0]
lines = (sourcefull.count("\n"))
commentareas = utils.findComments(sourcefull)

  
def pred(repo_code=sourcefull):
    line = 0
    file = open("../outputs/generator_function.txt", 'a')
    file.write(repo_code + "*********")
    retval = ''
    for char in repo_code:
        file.write("\nchar: " + char)
        retval += char if not char == '\n' else ''
        file.write("\n1 retval: " + retval)
        if char == '\n':
            file.write("\n2 retval: " +  retval)
            pred_file = open("../predictions/predictions_EXAMPLE_sql.txt", 'a')
            pred_file.write("line: " + str(line) + "\tcode: " + retval + "\tprediction: " + str(classifier_cp_1(retval)) + "\n")
            line += 1
            pred_file.close()
            yield retval
            retval = ''
     
    file.close()
    if retval:
        yield retval

file = open("../outputs/generator_function.txt", 'a') 
file.write(str(list(pred())))
file.close()
