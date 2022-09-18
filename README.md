# Data-Augmentation-Incremento-de-Datos
<aside>
🔹**Autor:** Franklin Hiustong Gutierrez Arizaca.

🔸**Docente:** Fred Torres Cruz.
 
❤Maestría en Ciencias - Ingeniería en Ingeniería de Sistemas.
</aside>

Las técnicas de aumento de datos se utilizan para aumentar la cantidad de datos añadiendo copias ligeramente modificadas de datos ya existentes o datos sintéticos recién creados a partir de datos existentes. Actúa como un regularizador y ayuda a reducir el sobreajuste cuando se entrena un modelo de aprendizaje automático. También está estrechamente relacionado con el sobremuestreo.

Fuente: [Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation)

En este cuaderno repasaremos las técnicas de aumento de datos más utilizadas, específicamente para datos de texto, y su implementación utilizando el paquete [Text-Data-Augmentation](https://github.com/Ritvik19/Text-Data-Augmentation).

```python
%%capture
!pip install sentencepiece
!pip install git+https://github.com/Ritvik19/Text-Data-Augmentation.git
#!python -m spacy download en_core_web_lg
!python -m spacy download es_core_news_lg
```

```python
%%capture
import nltk
nltk.download('all')
```

## **Abstractive Summarization (Abstracción de resumen)**

El aumento de la síntesis abstracta utiliza modelos de transformación de última generación para resumir el texto dado. [[17]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-17) [[18]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-18)

```python
from text_data_augmentation import AbstractiveSummarization
aug = AbstractiveSummarization()
aug(["""La Resumificación Abstractiva es una tarea del Procesamiento del Lenguaje Natural (PLN) que tiene como objetivo generar un resumen conciso de un texto fuente. A diferencia del resumen extractivo
 el resumen abstractivo no se limita a copiar las frases importantes del texto original, sino que también puede crear nuevas frases que sean relevantes, lo que puede considerarse como 
 paráfrasis. El resumen abstractivo tiene numerosas aplicaciones en diferentes ámbitos, desde los libros y la literatura hasta la ciencia y la I+D, pasando por la investigación financiera y el análisis de documentos jurídicos.
 análisis de documentos jurídicos."""])
```

## **Back Translation (Traducción Posterior)**

El aumento de la traducción inversa se basa en la traducción de datos textuales a otro idioma y su posterior traducción al idioma original. Esta técnica permite generar datos textuales de redacción distinta al texto original conservando el contexto y el significado originales.[[1]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-1) [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[10]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-10)

```python
from text_data_augmentation import BackTranslation
aug = BackTranslation()
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Character Noise (Ruido en el Carácter)**

El aumento del ruido de caracteres añade ruido a nivel de caracteres insertando, borrando, intercambiando o sustituyendo aleatoriamente algunos caracteres en el texto de entrada. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[9]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-9)

```python
from text_data_augmentation import CharacterNoise
aug = CharacterNoise(alpha=0.2, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Contextual Word Replacement (Sustitución de Palabras por Contexto)**

El Aumento Contextual de Reemplazo de Palabras crea Muestras Aumentadas sustituyendo aleatoriamente algunas palabras por una máscara y utilizando después un Modelo de Lenguaje Enmascarado para rellenarla. El muestreo de palabras también puede ponderarse utilizando valores TFIDF. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[3]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-3) [[11]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-11) [[19]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-19)

```python
from text_data_augmentation import ContextualWordReplacement
aug = ContextualWordReplacement(n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Easy Data Augmentation (Fácil Incremento de Datos)**

Easy Data Augmentation añade ruido a nivel de palabra insertando, borrando o intercambiando aleatoriamente algunas palabras en el texto de entrada o barajando las frases en el texto de entrada. [[4]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-4) [[5]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-5) [[9]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-9) [[12]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-12) [[13]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-13)

```python
from text_data_augmentation import EasyDataAugmentation
aug = EasyDataAugmentation(n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **KeyBoard Noise (Ruido del teclado)**

El Aumento del Ruido del Teclado añade ruido de errores ortográficos a nivel de caracteres imitando los errores tipográficos cometidos con un teclado qwerty en el texto de entrada. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[9]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-9)

```python
from text_data_augmentation import KeyBoardNoise
aug = KeyBoardNoise(alpha=0.1, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **OCR Noise (Ruido OCR)**

El aumento del ruido del OCR añade ruido de errores ortográficos a nivel de caracteres imitando los errores del OCR en el texto de entrada. [[6]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-6)

```python
from text_data_augmentation import OCRNoise
aug = OCRNoise(alpha=0.1, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Paraphrase (Parafraseo)**

El aumento de la paráfrasis reformula las frases de entrada utilizando modelos T5. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2)

```python
from text_data_augmentation import Paraphrase
aug = Paraphrase("hetpandya/t5-small-tapaco", n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Similar Word Replacement (Sustitución de palabras similares)**

El aumento de la sustitución de palabras similares crea muestras aumentadas sustituyendo aleatoriamente algunas palabras por una palabra que tenga el vector más similar a ella. El muestreo de palabras puede ponderarse utilizando también los valores TFIDF. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[7]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-7) [[15]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-15) [[16]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-16) [[19]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-19)

```python
from text_data_augmentation import SimilarWordReplacement
aug = SimilarWordReplacement("en_core_web_lg",  alpha=0.2, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Synonym Replacement (Sustitución de sinónimos)**

El aumento por sustitución de sinónimos crea muestras aumentadas sustituyendo aleatoriamente algunas palabras por sus sinónimos a partir de la base de datos de la red de palabras. El muestreo de palabras puede ponderarse utilizando también los valores TFIDF. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[4]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-4) [[8]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-8) [[13]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-13) [[19]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-19)

```python
from text_data_augmentation import SynonymReplacement
aug = SynonymReplacement(alpha=0.2, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## **Word Split (División de palabras)**

El aumento de la división de palabras añade ruido de error ortográfico a nivel de palabra dividiendo las palabras al azar en el texto de entrada. [[2]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-2) [[14]](https://colab.research.google.com/drive/1b5upH20DnCgdstm5NV973xFiP6U6SdQQ#ref-14)

```python
from text_data_augmentation import WordSplit
aug = WordSplit(alpha=0.15, n_aug=1)
aug(['A quick brown fox jumps over the lazy dog'])
```

## Referencias

1. [Data Expansion Using Back Translation and Paraphrasing for Hate Speech Detection](https://arxiv.org/pdf/2106.04681.pdf)
2. [A Survey on Data Augmentation for Text Classification](https://arxiv.org/ftp/arxiv/papers/2107/2107.03158.pdf)
3. [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/pdf/1805.06201.pdf)
4. [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)
5. [An Analysis of Simple Data Augmentation for Named Entity Recognition](https://aclanthology.org/2020.coling-main.343.pdf)
6. [Deep Statistical Analysis of OCR Errors for Effective Post-OCR Processing](https://zenodo.org/record/3245169/files/JCDL2019_Deep_Analysis.pdf)
7. [A Study of Various Text Augmentation Techniques for Relation Classification in Free Text](https://www.researchgate.net/publication/331784439_A_Study_of_Various_Text_Augmentation_Techniques_for_Relation_Classification_in_Free_Text)
8. [Text Augmentation for Neural Networks](http://ceur-ws.org/Vol-2268/paper11.pdf)
9. [Synthetic And Natural Noise Both Break Neural Machine Translation](https://arxiv.org/pdf/1711.02173.pdf)
10. [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709.pdf)
11. [Data Augmentation Using Pre-trained Transformer Models](https://arxiv.org/pdf/2003.02245.pdf)
12. [Data Augmentation via Dependency Tree Morphing for Low-Resource Languages](https://arxiv.org/pdf/1903.09460.pdf)
13. [Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models](https://arxiv.org/pdf/1809.02079.pdf)
14. [TextBugger: Generating Adversarial Text Against Real-world Applications](https://arxiv.org/pdf/1812.05271v1.pdf)
15. [Generating Natural Language Adversarial Examples](https://arxiv.org/pdf/1804.07998.pdf)
16. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)
17. [Neural Abstractive Text Summarization with Sequence-to-Sequence Models](https://arxiv.org/pdf/1812.02303.pdf)
18. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461v1.pdf)
19. [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848.pdf)
20. [Text Data Augmentation: Towards better detection of spear-phishing emails](https://arxiv.org/pdf/2007.02033.pdf)
