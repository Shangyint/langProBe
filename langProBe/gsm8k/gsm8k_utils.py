def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][
            -1
        ]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(
        parse_integer_answer(str(pred.answer))
    )


def gsm8k_evaluate(gold, pred, trace=None):
    gold_answer = int(gold.answer.split()[-1].replace(",", ""))
    return gold_answer == int(parse_integer_answer(str(pred.answer)))
