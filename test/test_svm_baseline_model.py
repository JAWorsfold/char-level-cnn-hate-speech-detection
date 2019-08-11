from src.svm_baseline_model import *

#test data
#Need to know what the stemmer will return. Use test tweets (obama etc).

input_one = "\"No one is born hating another person because of the color of his skin or his background or his religion...\"   https://t.co/InZ58zkoAm"
process_one = "\"No one is born hating another person because of the color of his skin or his background or his religion...\" "
token_one = "no one is born hate anoth person becaus of the color of hi skin or hi background or hi religion   https://t.co/InZ58zkoAm"
both_one = "no one is born hate anoth person becaus of the color of hi skin or hi background or hi religion"

input_two = "Our visit to @projectaware discussing the progress we made together and exploring future collaborations with @epa_kw highlighting its role in supporting environmental voluntary initiatives #guardiansofthesea https://www.instagram.com/p/B1A4NFCgbmo/?igshid=1vc2qn16t2spd"
process_two = "Our visit to discussing the progress we made together and exploring future collaborations with highlighting its role in supporting environmental voluntary initiatives "
token_two = "our visit to @projectaware discuss the progress we made togeth and explor futur collabor with @epa_kw highlight it role in support environment voluntari initi #guardiansofthesea https://www.instagram.com/p/B1A4NFCgbmo/?igshid=1vc2qn16t2spd"
both_two = "our visit to discuss the progress we made togeth and explor futur collabor with highlight it role in support environment voluntari  "

input_three = "Empty food dish.. pretty sure that look he is giving me is saying something.. you think @thechrisbarron? #Caturday"
process_three = "Empty food dish.. pretty sure that look he is giving me is saying something.. you think "
token_three = "empti food dish pretti sure that look he is give me is say something you think @thechrisbarron? #Caturday"
both_three = "empti food dish pretti sure that look he is give me is say something you think"

input_four = "TensorWatch: A debugging and visualization system for machine learning http://bit.ly/2KFUvqe   #AI   #DeepLearning   #MachineLearning  #DataScience"
process_four = "TensorWatch: A debugging and visualization system for machine learning "
token_four = "tensorwatch a debug and visual system for machin learn http://bit.ly/2KFUvqe   #ai   #deeplearn   #machinelearn  #dataSci"
both_four = "tensorwatch a debug and visual system for machin learn"


def test_preprocessor():
    actual_one = preprocessor(input_one)
    actual_two = preprocessor(input_two)
    actual_three = preprocessor(input_three)
    actual_four = preprocessor(input_four)
    assert True


def test_tokenizer():
    actual_one = tokenizer(input_one)
    actual_two = tokenizer(input_two)
    actual_three = tokenizer(input_three)
    actual_four = tokenizer(input_four)
    assert True


def test_tokenize_and_preprocess():
    actual_one = tokenizer(preprocessor(input_one))
    actual_two = tokenizer(preprocessor(input_two))
    actual_three = tokenizer(preprocessor(input_three))
    actual_four = tokenizer(preprocessor(input_four))
    assert True
