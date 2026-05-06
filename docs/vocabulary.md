# SignLearn Vocabulary

## Alphabet (26)

- a
- b
- c
- d
- e
- f
- g
- h
- i
- j
- k
- l
- m
- n
- o
- p
- q
- r
- s
- t
- u
- v
- w
- x
- y
- z

## Digits (10)

- zero
- one
- two
- three
- four
- five
- six
- seven
- eight
- nine

## Static Words

### Greetings
- hello
- goodbye
- good_morning
- good_afternoon
- good_night
- welcome

### Confirmations & Responses
- yes
- no
- okay
- correct
- wrong
- maybe
- agree
- disagree

### Basic Interactions
- please
- thank_you
- sorry
- excuse_me
- help
- understand
- not_understand
- repeat
- finish
- more

## Dynamic Words

### Questions
- what
- where
- when
- who
- why
- how
- how_much
- how_many

### Emergency Terms
- stop
- danger
- hospital
- doctor
- pain
- sick
- call_police
- fire

### Daily Actions
- eat
- drink
- go
- come
- work
- sleep
- drive
- read
- write
- run
- sit
- stand

### Social Terms
- friend
- family
- love
- meet
- talk

---

## Notes

- **ML Classification Design**: Every label in this vocabulary is a discrete, unambiguous class identifier intended for use as the output layer of a sign language recognition model. The flat, snake_case format ensures direct compatibility with standard ML frameworks (TensorFlow, PyTorch, scikit-learn) without requiring any pre-processing or label encoding transformations.
- **Future Model Outputs**: Labels are designed to serve as the final predicted class names emitted by a trained classifier. Downstream applications (e.g., real-time captioning, mobile UI) can consume these labels directly without additional mapping layers.
- **Synonym Avoidance**: Similar terms (e.g., `hello` only, not `hi`) are intentionally excluded to prevent class ambiguity during training and reduce inter-class confusion in the model's decision boundary.
- **Category Rationale**: Static words represent signs that can be reliably classified from a single frame or short clip, while dynamic words require temporal modeling (e.g., LSTMs, transformers, or optical flow) due to motion-dependent articulation.
- **Total Label Count**: 93 unique labels (26 alphabet + 10 digits + 24 static + 33 dynamic).
