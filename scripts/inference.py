from transformers import pipeline
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