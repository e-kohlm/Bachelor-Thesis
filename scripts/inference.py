from transformers import pipeline
import json
import VUDENC_utils
#from fine_tuning import args.save_dir
  
# Inference

#https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline
#https://huggingface.co/docs/transformers/task_summary#sequence-classification
#TODO https://huggingface.co/docs/evaluate/a_quick_tour


print("\nInference")
# TODO load_best_model_at_end=true was macht das? ist aus tutorial_sequence_classification , den checkpoint musste ich trotzdem in Pfad packen
# TODO ohne checkpoint wird keine config.json gefunden
# TODO Dynamisch machen, nicht jeden checkpoint einzeln
# TODO final checkpoint hat nicht die notwendigen Daten für Inference, warum? 


classifier_cp_1 = pipeline(task="text-classification", model="saved_models/summarize_python" + "/checkpoint-280")
#classifier_f_cp = pipeline(task="text-classification", model=args.save_dir + "/final_checkpoint") # Dateien fehlen

test_text = "tokenized_datasets = tokenized_datasets.remove_columns(['snippet_id']) tokenized_datasets = tokenized_datasets.rename_column('label', 'labels') tokenized_datasets.set_format('torch')"
vul_snippet = "SQL_RECURSIVE_QUERY_EDUCATION_GROUP='''\\ WITH RECURSIVE group_element_year_parent AS( SELECT id, child_branch_id, child_leaf_id, parent_id, 0 AS level FROM base_groupelementyear WHERE parent_id IN({list_root_ids'"
not_vul_snippet = "' INNER JOIN group_element_year_parent AS parent on parent.child_branch_id=child.parent_id ) SELECT * FROM group_element_year_parent ; ''''''''' class GroupElementYearManager(models.Manager): def get_queryset"
# TODO kein snippet reingeben, sondern viel Code, was passiert dann damit?


print("test_text_cp_1: ", classifier_cp_1(test_text))
#print("test text_f_cp: ", classifier_f_cp(test_text))
print("vul_cp_1: ", classifier_cp_1(vul_snippet))
print("not_vul_cp 1: ", classifier_cp_1(not_vul_snippet))


print(" Github code reingeben ##############")

# TODO: Verstehen: der Text wird hier nicht tokenized, bzw. verstehen was da genau im Hintergrund passiert, auch wenn ich hier keinen tokenizer calle

# aus demonstrate.py
mode = 'sql'
nr = '1'

rep = ""
com = ""
myfile = ""

with open('../VUDENC_data/plain_' + mode, 'r') as infile:  # Warum lädt sie denn hier den file, der schon zum trainieren des models genutzt wurde?
  data = json.load(infile)                       # nur zu demozwecken denke ich

identifying = VUDENC_utils.getIdentifiers(mode,nr)  # hier wird github repo ausgewählt, das aus vul getestet werden soll
info = VUDENC_utils.getFromDataset(identifying,data) # data ist wieder repo und commit von datei, mit der ich trainiert habe
                                                # es wird zu demozwecken nur das erste brauchbare repo bzw. commit geholt
#print("info: ", info)                           # info ist nur 1 commit mit changes = demonstrate_get_from_dataset.txt
sourcefull = info[0]
print("sourcefull:\n", sourcefull)
print("type: ", type(sourcefull))
lines = (sourcefull.count("\n"))
print("lines: ", lines)
commentareas = VUDENC_utils.findComments(sourcefull) # gibt liste mit anfangspos und endpos von einzeikigen # kommentaren zurück
print("commentareas: ", commentareas) # rausnehmen, 
  
def f2(repo_code=sourcefull):
    line = 0
    file = open("../test_outputs/generator_function.txt", 'a')
    file.write("f2 is used\n" + repo_code + "*********")
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
file = open("../test_outputs/generator_function.txt", 'a') ##Achtung überschreibt nicht, file wird länger und länger
file.write(str(list(f2())))
file.close()

# TODO: function is ugly as hell und verstehen tu ich das auch nicht richtig, mit yield und der generator function
