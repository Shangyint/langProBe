import string


def preprocess_text(text):
    # Remove all punctuation
    text = "".join(char for char in text if char not in string.punctuation)
    # Convert to lower case
    text = text.lower()
    # Remove 'peace!' from the end if it exists
    if text.endswith("peace!"):
        text = text[
            :-6
        ].strip()  # Remove the last 'peace!' and strip any trailing spaces
    return text

def check_conditions(example, pred, trace=None, debug=False):
    category = example.category
    answer = pred.answer

    preprocessed_pred_answer = preprocess_text(pred.answer)
    preprocessed_correct_answer = preprocess_text(example.answer)

    # Check for exact match after preprocessing
    if preprocessed_pred_answer != preprocessed_correct_answer:
        if debug:
            print(
                f"Exact match failed. Expected: '{preprocessed_correct_answer}', Got: '{preprocessed_pred_answer}'"
            )
        return False

    # When the answer is a place, the response should contain no punctuation
    if category == "place":
        if any(char in answer for char in ",.?!;:"):
            return False
        else:
            if debug:
                print(
                    f"Place. When the answer is a place, the response should contain no punctuation {answer}"
                )
            return True

    # When the answer is a date, the response should end with "Peace!"
    elif category == "date":
        if not answer.endswith("Peace!"):
            return False
        else:
            if debug:
                print(
                    f"Date. When the answer is a date, the response should end with Peace! {answer}"
                )
            return True

    # When the answer is a person, the response should be entirely in lowercase
    elif category == "person":
        if answer != answer.lower():
            return False
        else:
            if debug:
                print(
                    f"Answer. When the answer is a person, the response should be entirely in lowercase {answer}"
                )
            return True

    # When the answer is none of the above categories, the response should be in all caps and not end with "Peace!"
    else:
        if answer != answer.upper() or answer.endswith("Peace!"):
            return False
        else:
            if debug:
                print(
                    f"Other category. the response should be in all caps and not end with Peace! {answer}"
                )
            return True