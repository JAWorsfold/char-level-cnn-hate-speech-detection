from src.svm_baseline_model import *

#test data
#Need to know what the stemmer will return. Use test tweets (obama etc).

input_one = "\"No one is born hating another person because of the color of his skin or his background or his religion...\"   https://t.co/InZ58zkoAm"
process_one = "\"No one is born hating another person because of the color of his skin or his background or his religion...\" "
token_one = "no one is born hate anoth person becaus of the color of hi skin or hi background or hi religion httpstcoinzzkoam"
both_one = "no one is born hate anoth person becaus of the color of hi skin or hi background or hi religion"

input_two = "Our visit to @projectaware discussing the progress we made together and exploring future collaborations with @epa_kw highlighting its role in supporting environmental voluntary initiatives #guardiansofthesea https://www.instagram.com/p/B1A4NFCgbmo/?igshid=1vc2qn16t2spd"
process_two = "Our visit to discussing the progress we made together and exploring future collaborations with highlighting its role in supporting environmental voluntary initiatives #guardiansofthesea "
token_two = "our visit to projectawar discuss the progress we made togeth and explor futur collabor with epakw highlight it role in support environment voluntari initi guardiansofthesea httpswwwinstagramcompbanfcgbmoigshidvcqntspd"
both_two = "our visit to discuss the progress we made togeth and explor futur collabor with highlight it role in support environment voluntari initi guardiansofthesea"

input_three = "Empty food dish.. pretty sure that look he is giving me is saying something.. you think @thechrisbarron? #Caturday &#123456;"
process_three = "Empty food dish.. pretty sure that look he is giving me is saying something.. you think ? #Caturday "
token_three = "empti food dish pretti sure that look he is give me is say someth you think thechrisbarron caturday"
both_three = "empti food dish pretti sure that look he is give me is say someth you think caturday"

input_four = "TensorWatch: A debugging and visualization system for machine learning http://bit.ly/2KFUvqe &amp;  #AI   #DeepLearning   #MachineLearning  #DataScience"
process_four = "TensorWatch: A debugging and visualization system for machine learning #AI #DeepLearning #MachineLearning #DataScience"
token_four = "tensorwatch a debug and visual system for machin learn httpbitlykfuvq amp ai deeplearn machinelearn datasci"
both_four = "tensorwatch a debug and visual system for machin learn ai deeplearn machinelearn datasci"


def test_svm_preprocessor():
    actual_one = svm_preprocessor(input_one)
    actual_two = svm_preprocessor(input_two)
    actual_three = svm_preprocessor(input_three)
    actual_four = svm_preprocessor(input_four)
    assert process_one == actual_one
    assert process_two == actual_two
    assert process_three == actual_three
    assert process_four == actual_four


def test_svm_tokenizer():
    actual_one = svm_tokenizer(input_one)
    actual_two = svm_tokenizer(input_two)
    actual_three = svm_tokenizer(input_three)
    actual_four = svm_tokenizer(input_four)
    assert token_one == ' '.join(actual_one)
    assert token_two == ' '.join(actual_two)
    assert token_three == ' '.join(actual_three)
    assert token_four == ' '.join(actual_four)


def test_tokenize_and_preprocess():
    actual_one = ' '.join(svm_tokenizer(svm_preprocessor(input_one)))
    actual_two = ' '.join(svm_tokenizer(svm_preprocessor(input_two)))
    actual_three = ' '.join(svm_tokenizer(svm_preprocessor(input_three)))
    actual_four = ' '.join(svm_tokenizer(svm_preprocessor(input_four)))
    assert both_one == actual_one
    assert both_two == actual_two
    assert both_three == actual_three
    assert both_four == actual_four
