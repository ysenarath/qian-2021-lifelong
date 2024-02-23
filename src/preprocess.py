import preprocessor
import re
import string

__all__ = [
    "tweet_preprocessor",
]


def tweet_preprocessor(
    text: str,
    rmurl: bool = True,
    rmreserved: bool = True,
    rmemoji: bool = False,
    rmhashtag: bool = False,
    rmpunc: bool = False,
    rpmention: bool = False,
    lowercase: bool = False,
):
    if rmurl:
        preprocessor.set_options(preprocessor.OPT.URL)
        text = preprocessor.clean(text)
    if rmreserved:
        preprocessor.set_options(preprocessor.OPT.RESERVED)
        text = preprocessor.clean(text)
    if rmemoji:
        preprocessor.set_options(preprocessor.OPT.EMOJI)
        text = preprocessor.clean(text)
    if rmhashtag:
        preprocessor.set_options(preprocessor.OPT.HASHTAG)
        text = preprocessor.clean(text)
    if rpmention:
        text = re.sub(r"@\S+", "@USER", text)
    if rmpunc:
        mypunc = list(string.punctuation)
        mypunc.remove("@")
        mypunc.remove("#")
        mypunc = "".join(mypunc)
        translator = str.maketrans("", "", mypunc)
        text = text.translate(translator)
    if lowercase:
        text = text.lower()
    return text
