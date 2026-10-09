# Project Update Checklist

## 1. Backend stability
- [x] Replace the fragile search code with a safer version
- [x] Ensure the API does not crash when the dataset is missing
- [x] Add working text and intelligent search endpoints

## 2. Dataset strategy
- [x] Support local JSON/CSV product catalogs
- [x] Add a safe demo catalog fallback for delivery
- [ ] Prepare a cleaned local furniture catalog if your own data is usable
- [ ] Use a public furniture/home dataset if your local data is too noisy

## 3. Retrieval quality
- [ ] Add a re-ranking step after initial retrieval
- [ ] Add product attribute extraction such as color/material/style
- [ ] Add visual grounding to reduce background noise in image search

## 4. Evaluation
- [ ] Add Recall@K, Precision@K, and NDCG reporting
- [ ] Add a small user-study evaluation if time allows

## 5. Delivery checklist
- [ ] Test the /health endpoint
- [ ] Test /search/text
- [ ] Test /search/intelligent
- [ ] Prepare demo queries and screenshots
- [ ] Save the final version for submission