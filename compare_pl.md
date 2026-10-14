# EN->PL FLORES comparison (25 rows, devtest)

## chrF vs human Polish reference

| model | variant-0 chrF | mean of per-variant chrF |
|---|---|---|
| Qwen3.5-2B | 41.77 | 41.77 |
| Qwen2.5-3B-Instruct | 37.86 | 37.86 |

## throughput

| model | tokens | gen seconds | tok/s |
|---|---|---|---|
| Qwen3.5-2B | 8620 | 87.5 | 98.48 |
| Qwen2.5-3B-Instruct | 9476 | 136.4 | 69.46 |

## examples (paraphrase->translate)

### 1
- **EN**: "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
- **human PL**: „Mamy teraz myszy w wieku 4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.
- **Qwen3.5-2B**: Wyglądający na mówiącego teraz przedstawia 4-miesięczne myszy, które wcześniej miały cukrzycę i od tego czasu stały się niecukrzyste.
- **Qwen2.5-3B-Instruct**: Oni wspomnieni również, że 'obecnie mamy myszy o 4-miesięcznej karczmei, które były oryginalnie diabetyczne, ale obecnie nie są diabetyczne.'

### 2
- **EN**: Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.
- **human PL**: Dr Ehud Ur, będący profesorem medycyny na Uniwersytecie Dalhousie w Halifaxie w Nowej Szkocji oraz przewodniczącym oddziału klinicznego i naukowego Kanadyjskiego Stowarzyszenia Cukrzycy, przestrzegł, iż badania nadal dopiero się zaczynają.
- **Qwen3.5-2B**: Dr. Ehud Ur, profesor medycyny w Dalhousie University w Halifax, w Nowej Szkocji, i przewodniczący klinicznej i naukowej sekcji Canadian Diabetes Association, zaznaczył, że badanie znajduje się obecnie w fazie początkowej.
- **Qwen2.5-3B-Instruct**: Prof. Ehud Ur, lekarz z Dalhousie University w Halifax, Nova Scotia, i przewodniczący oddziału kliniczno-technicznego Związku Diabetycznego Kanady, zaznaczył, że badania są nadal w fazie początkowej.

### 3
- **EN**: Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.
- **human PL**: Tak jak część ekspertów jest sceptyczny co do tego, czy cukrzycę da się wyleczyć; zauważa, że te ustalenia są bez znaczenia dla osób, którze już chorują na cukrzycę typu 1.
- **Qwen3.5-2B**: Tak jak utrzymują pewni eksperci, autor kwestionuje możliwość leczenia cukrzycy, podkreślając, że wyniki nie dotyczą osób cierpiących na cukrzycę typu 1.
- **Qwen2.5-3B-Instruct**: W ten sam sposób wyraża pewne ekspertyzy nieufność wobec możliwości leczenia cukrzycy, zaznaczając, że te wyniki nie są zastosowane do osób, które już mają cukrzycę typu 1.

### 4
- **EN**: On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.
- **human PL**: W poniedziałek Sara Denius, stała sekretarz Komitetu Noblowskiego ds. Literatury przy Szwedzkiej Akademii, w czasie programu radiowego w szwedzkim Sveriges Radio, podała publicznie informację, iż komisja zaprzestała już prób nawiązania bezpośredniego kontaktu z Bobem Dylanem w kwestii zdobycia przez niego Literackiej Nagrody Nobla w roku 2016.
- **Qwen3.5-2B**: W poniedziałek Sara Danius, stała sekretarz generalna Komitetu Nobla poezji w Akademii Szwajcarskiej, podczas transmisji radiowej na Sveriges Radio ujawniła, że komitet zrezygnował z prób kontaktu z Bobem Dylanem po tym, jak nie udało się dotrzeć do niego w sprawie jego kandydatury na Nagrodę Nobla w dziedzinie literatury w 2016 roku.
- **Qwen2.5-3B-Instruct**: W poniedziałek, Sara Danius, sekretarz stowarzyszenia Nobel dla literatury na Szwedzkiej Akademii, podczas programu radiowego na Sveriges Radio, disclose, że komite wydał się niepotrzebne próbę kontaktowania Boba Dylan bezpośrednio, aby informować go o nagrodzie nagrody Nobel Literatury za 2016 roku, z powodu ich niepowodzenia w dołączeniu do niego.
