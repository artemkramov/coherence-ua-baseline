# Baseline coherence estimation model

Baseline coherence estimation model for the Ukrainian language. The model is based on the usage of ELmo embeddings for
words' encoding; the architecture of the model is based on LSTM layers as it was stated in the corresponding
article: https://link.springer.com/chapter/10.1007/978-3-031-14841-5_33.

## Installation

The module is mainly based on the tensorflow library (see `requirements.txt` for more details). As far as the module is
based on other external tools as well, the following files should be downloaded and placed:

- [ELmo weights](https://www.googleapis.com/drive/v3/files/1Vk9xiJs86Cy04sjhQkX87lJNDSpExoeT?alt=media&key=AIzaSyDoZ9wyoY8M-5_MMZxiqwAwVSI4IV8KAxo)
  should be placed into the folder `model_elmo`;
- [UDpipe model weights](https://www.googleapis.com/drive/v3/files/1TN4pHHQuNJT1DwppvWs94ZYzeIvynHjs?alt=media&key=AIzaSyDoZ9wyoY8M-5_MMZxiqwAwVSI4IV8KAxo)
  that were used for the preprocessing of the input Ukrainian text should be placed in the folder `ufal`;
- [Model weights](https://www.googleapis.com/drive/v3/files/1-ab9Umf6QOh7TTQA23s7CSfbULxIdkiE?alt=media&key=AIzaSyDoZ9wyoY8M-5_MMZxiqwAwVSI4IV8KAxo)
  should be placed in the folder `model_lstm`.

## Usage

Import the class `BaselineCoherence` from the file `baseline.py` and run the function `evaluate_coherence_logscore`.

Code example:

```python
from baseline import BaselineCoherence

baseline = BaselineCoherence()

text = """Науковці відкрили ще 62 нові супутники у Сатурна, і це повернуло йому статус планети з їхньою найбільшою кількістю.

Кілька місяців тому лідером у Сонячній системі був газовий гігант Юпітер, повідомляє The Guardian.

Загальна кількість супутників Сатурна становить 145, а Юпітера – 95. Це офіційно визначив Міжнародний астрономічний союз (IAU).

"Сатурн не тільки майже подвоїв кількість супутників, але й тепер у нього їх більше, ніж у всіх інших планет Сонячної системи разом узятих", –  сказав професор, астроном з Університету Британської Колумбії Бретт Гладман. 
"""

score = baseline.evaluate_coherence_logscore(text)

print(score)

"""
-0.8644944
"""

```

The function `evaluate_coherence_logscore` takes the following parameter:

- `text` - `str`, an input Ukrainian text

The function `evaluate_coherence_logscore` returns the float value in the
  range `[-inf, 0]` where a lower value indicates better coherence measure.
