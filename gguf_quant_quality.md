# GGUF quantization translation quality (EN->IT, FLORES)

- data: `flores200_en_it_de_fr.csv` (latin-1), first 150 sentences
- prompt: `crosslingual` (`paraphrase_prompts.build_messages`), model chat template
- decoding: greedy (temp 0) until EOS, full GPU offload (n_gpu_layers=-1)
- metrics: sacrebleu chrF (char_order 6, word_order 0), chrF++ (word_order 2), BLEU

| quant | size MB | chrF | chrF++ | BLEU | avg tok/s |
|---|---|---|---|---|---|
| Q4_K_M | 1281 | 53.62 | 50.46 | 22.56 | 66.1 |
| Q8_0 | 2012 | 53.80 | 50.60 | 22.53 | 55.7 |
| BF16 | 3776 | 54.01 | 50.81 | 22.74 | 42.4 |

## Examples

### Q4_K_M
- **EN**: On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.
  - **hyp**: Il 1° lunedì, ricercatori della Stanford University School of Medicine hanno annunciato l'invenzione di un nuovo strumento diagnostico in grado di classificare le cellule per tipo: un chip microscopico stampabile che può essere prodotto con una stampante a inchiostro standard per circa un centesimo di dollaro per unità.
  - **ref**: Nella giornata di lunedÃ¬, alcuni scienziati della Scuola di Medicina dell'UniversitÃ  di Stanford hanno annunciato l'invenzione di un nuovo strumento diagnostico capace di ordinare le cellule in base al tipo: un chip minuscolo che puÃ² essere stampato utilizzando stampanti a getto di inchiostro al costo di circa 1 centesimo di dollaro l'uno.
- **EN**: Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.
  - **hyp**: I ricercatori principali affermano che questo potrebbe portare alla diagnosi precoce del cancro, della tubercolosi, dell'HIV e della malaria nei paesi a basso reddito, dove le tassi di sopravvivenza per malattie come il cancro al seno possono essere la metà di quelli dei paesi più ricchi.
  - **ref**: Secondo i responsabili della ricerca questo strumento potrebbe consentire diagnosi precoci di cancro, tubercolosi, HIV e malaria per i malati che vivono in Paesi a basso reddito. In alcuni di questi Paesi il tasso di sopravvivenza legati a patologie come il cancro al seno puÃ² essere addirittura la metÃ  rispetto a quello registrato nei Paesi piÃ¹ abbienti.
- **EN**: The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.
  - **hyp**: Il JAS 39C Gripen ha colpito una pista a circa le 9:30 del mattino locale (0230 UTC) e ha esploduto, chiudendo l'aeroporto ai voli commerciali.
  - **ref**: Il JAS 39C Gripen Ã¨ precipitato su una pista di atterraggio alle 9:30 circa, ora locale (02:30 UTC), ed Ã¨ esploso, con la conseguente chiusura dellâaeroporto ai voli commerciali.

### Q8_0
- **EN**: On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.
  - **hyp**: Il lunedì, ricercatori della Stanford University School of Medicine hanno annunciato l'invenzione di un nuovo strumento diagnostico in grado di classificare le cellule per tipo: un chip di stampa microscopico che può essere prodotto utilizzando stampanti a inchiostro standard per circa un centesimo di dollaro a pezzo.
  - **ref**: Nella giornata di lunedÃ¬, alcuni scienziati della Scuola di Medicina dell'UniversitÃ  di Stanford hanno annunciato l'invenzione di un nuovo strumento diagnostico capace di ordinare le cellule in base al tipo: un chip minuscolo che puÃ² essere stampato utilizzando stampanti a getto di inchiostro al costo di circa 1 centesimo di dollaro l'uno.
- **EN**: Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.
  - **hyp**: I ricercatori diretti affermano che questo potrebbe portare alla rilevazione precoce del cancro, della tubercolosi, dell'HIV e della malaria nei paesi a basso reddito, dove le tassi di sopravvivenza per malattie come il cancro al seno possono essere la metà di quelli dei paesi più ricchi.
  - **ref**: Secondo i responsabili della ricerca questo strumento potrebbe consentire diagnosi precoci di cancro, tubercolosi, HIV e malaria per i malati che vivono in Paesi a basso reddito. In alcuni di questi Paesi il tasso di sopravvivenza legati a patologie come il cancro al seno puÃ² essere addirittura la metÃ  rispetto a quello registrato nei Paesi piÃ¹ abbienti.
- **EN**: The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.
  - **hyp**: Il JAS 39C Gripen è atterrato su una pista intorno alle 9:30 del mattino (0230 UTC) e ha esploduto, chiudendo l'aeroporto ai voli commerciali.
  - **ref**: Il JAS 39C Gripen Ã¨ precipitato su una pista di atterraggio alle 9:30 circa, ora locale (02:30 UTC), ed Ã¨ esploso, con la conseguente chiusura dellâaeroporto ai voli commerciali.

### BF16
- **EN**: On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.
  - **hyp**: Il lunedì, ricercatori dell'Università Stanford School of Medicine hanno annunciato l'invenzione di un nuovo strumento diagnostico in grado di classificare le cellule per tipo: un chip di stampa microscopico che può essere prodotto utilizzando stampanti a inchiostro standard per circa un centesimo di dollaro a pezzo.
  - **ref**: Nella giornata di lunedÃ¬, alcuni scienziati della Scuola di Medicina dell'UniversitÃ  di Stanford hanno annunciato l'invenzione di un nuovo strumento diagnostico capace di ordinare le cellule in base al tipo: un chip minuscolo che puÃ² essere stampato utilizzando stampanti a getto di inchiostro al costo di circa 1 centesimo di dollaro l'uno.
- **EN**: Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.
  - **hyp**: I ricercatori principali affermano che questo potrebbe portare alla rilevazione precoce del cancro, della tubercolosi, dell'HIV e della malaria nei paesi a basso reddito, dove le tassi di sopravvivenza per malattie come il cancro al seno possono essere la metà di quelli dei paesi più ricchi.
  - **ref**: Secondo i responsabili della ricerca questo strumento potrebbe consentire diagnosi precoci di cancro, tubercolosi, HIV e malaria per i malati che vivono in Paesi a basso reddito. In alcuni di questi Paesi il tasso di sopravvivenza legati a patologie come il cancro al seno puÃ² essere addirittura la metÃ  rispetto a quello registrato nei Paesi piÃ¹ abbienti.
- **EN**: The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.
  - **hyp**: Il JAS 39C Gripen è atterrato su una pista intorno alle 9:30 del mattino (0230 UTC) e ha esploduto, chiudendo l'aeroporto ai voli commerciali.
  - **ref**: Il JAS 39C Gripen Ã¨ precipitato su una pista di atterraggio alle 9:30 circa, ora locale (02:30 UTC), ed Ã¨ esploso, con la conseguente chiusura dellâaeroporto ai voli commerciali.
