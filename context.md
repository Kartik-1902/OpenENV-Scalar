I am competing in the openenv hackathon by meta x scaler 
below is my idea and link to github repo that we are given
1) Dataset columns are: -
ID, Date, Long term context, short term context, Stock predictions, Difficulty level
ID: Serial number of dataset
Date: Date of the dataset entry
Long term context: Take 1 week range in news api with sort by relevancy and take the top 10 articles.
Short term context: Take a day range according to Date with sort by popularity and take top 10 articles. Take one article for one datapoint. *So 10 datapoints on a single date*.
Stock predictions: Use yfinance to take top 5 gainers and losers of Date and a few chosen assets like gold, silver. Gainers & losers should be a ranked list and assets changes should only be on gain or loss, *LLM MUST NOT PREDICT NUMBER, JUST WHAT WILL BE AFFECTED*.
Difficulty level: Easy if direct reference to a stock and it's movement (Oil moved up). Medium if reference but no movement (Oil reserve was affected today). Hard if no reference (War happened, oil moved)
2)Short term context: All 10 datapoints on a single data.
Ab difficulty level kaise karna hai thora dekhle
3) Wordle kinda hi reward function bana lo based on asset appearing on list and appearing on list & in correct position too
4) https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md
Dataset:
Api me keywords ka parameter hai, to usme stocks aur assets like gold, silver, rare earth metals, oil use karlo
Plus badi news ke liye - War, scarcity, demand, supply, etc, jaise keywords nikal dete hai. GPT se ek list banwa lenge
from & to me dates daal sakte hai
 domains waale parameter me trusted & popular domains se hi filter karenge, jaise BBC, CNBC, jo bhi worldwide reputed news provider hai
yfinance api se kisi bhi din ke stock and asset results mil jaayenge
Sab python me ho jaayea https://gemini.google.com/share/fb362bd6eb7e
 https://newsapi.org/docs/client-libraries/python
you are an expert in AI/ML give me a full proff plan on architechture we have to use llm to predict.
Show more
18:10
Architected comprehensive ML pipeline integrating news APIs and financial data
Architected comprehensive ML pipeline integrating news APIs and financial data
This is a well-scoped and genuinely interesting hackathon project — essentially building a news-driven RL benchmark for financial reasoning. Here's the full architecture plan.

Project overview
You're building two things: (1) a dataset generation pipeline that pairs news context with verified stock outcomes, and (2) an OpenEnv RL environment where an LLM agent is trained to predict those outcomes using a Wordle-style reward. The LLM never predicts numbers — only which assets move and in which direction.

Architecture — full pipeline

Phase 1 — Dataset generation pipeline
NewsAPI query strategy. Use two separate calls per date:
Long-term context: from = date - 7 days, to = date, sortBy = relevancy, pageSize = 10. Keywords should combine asset terms ("gold", "oil", "silver", "rare earth") with macro triggers ("war", "sanctions", "supply chain", "rate hike", "inflation"). Get GPT to generate a master keyword list of ~40 terms once and hardcode it.
Short-term context: from = date, to = date, sortBy = popularity, pageSize = 10. Each article becomes one row in the dataset, so you get exactly 10 rows per date.
Domains filter: domains = "bbc.co.uk,cnbc.com,reuters.com,bloomberg.com,ft.com,wsj.com,apnews.com" — this keeps quality high and prevents SEO spam from polluting training data.
yfinance ground truth. Pull the top 5 gainers and losers for the Date using the S&P 500 ticker list. For fixed assets, compute pct_change for GLD, SLV, USO, PDBC (rare earth proxy). Store as:
Gainers: ["NVDA", "META", "XOM", ...] (ordered, rank matters for reward)
Losers: ["SVB", "INTC", ...]
Assets: {"gold": "UP", "silver": "DOWN", "oil": "UP"}
Difficulty labeller. Run GPT once per row with a prompt like: "Given this article headline and the following stocks that moved today, classify as Easy (article directly names the stock + direction), Medium (article references the sector/commodity but not the direction), or Hard (article is a macro event with no direct reference to these stocks)." Cache labels so you don't re-call.

Phase 2 — Dataset schema
ID | Date | long_term_context | short_term_article | stock_predictions | difficulty
stock_predictions is a JSON object:
json
{
  "gainers": ["NVDA", "XOM", "META", "AAPL", "CVX"],
  "losers":  ["INTC", "PFE", "C", "BA", "WBA"],
  "assets":  {"gold": "UP", "silver": "NEUTRAL", "oil": "UP"}
}
The LLM must output the same schema — no numbers, just tickers and direction strings.

Phase 3 — OpenEnv integration
Looking at the 04-training.md structure, your env class should look like:
python
class NewsStockEnv:
    def reset(self, idx):
        row = dataset[idx]
        self.ground_truth = row["stock_predictions"]
        self.difficulty = row["difficulty"]
        state = f"""
        Long-term context (past week): {row['long_term_context']}
        Today's article: {row['short_term_article']}
        Task: Predict top 5 gainers, top 5 losers (ranked), and direction for gold/silver/oil.
        Output JSON only.
        """
        return state

    def step(self, action):  # action = LLM's JSON output
        reward = compute_wordle_reward(action, self.ground_truth)
        difficulty_multiplier = {"Easy": 1.0, "Medium": 1.5, "Hard": 2.0}
        return reward * difficulty_multiplier[self.difficulty]
The difficulty multiplier is a simple but powerful trick — it up-weights harder reasoning tasks during RL training without changing the reward structure.

Phase 4 — LLM agent
Use Llama 3.1 8B (or Llama 3.2 if Meta provides access). Key prompting decisions:
System prompt must be strict: "You are a financial analyst. Output ONLY valid JSON matching this schema: {...}. Never output numbers or price targets."
Chain-of-thought scratchpad: allow the model an internal <think> block before the final JSON (this helps reasoning without polluting the output format). OpenEnv supports this natively with the thinking turn.
At inference, parse the output JSON and pass it to step().
For RL training: use GRPO (Group Relative Policy Optimization) which OpenEnv is built around. The reward signal from your Wordle function plugs in directly as the scalar reward.

Phase 5 — Wordle reward function
python
def compute_wordle_reward(prediction, ground_truth):
    reward = 0.0

    # Gainers list (ranked)
    for rank, ticker in enumerate(prediction.get("gainers", [])):
        if ticker in ground_truth["gainers"]:
            true_rank = ground_truth["gainers"].index(ticker)
            reward += 2.0 if true_rank == rank else 1.0

    # Losers list (ranked) — same logic
    for rank, ticker in enumerate(prediction.get("losers", [])):
        if ticker in ground_truth["losers"]:
            true_rank = ground_truth["losers"].index(ticker)
            reward += 2.0 if true_rank == rank else 1.0

    # Fixed assets — direction only
    for asset in ["gold", "silver", "oil"]:
        if prediction.get("assets", {}).get(asset) == ground_truth["assets"].get(asset):
            reward += 1.0

    return reward  # max possible ≈ 23 (10×2 + 3×1)
Normalise by difficulty so the agent gets consistent gradient signal.

What makes this project stand out at the hackathon
The insight that the reward function should be positional (Wordle-style) rather than binary is the core innovation — it gives a much denser signal than "did you get it right". The three-tier difficulty labelling means you can report breakdown metrics (Easy/Medium/Hard accuracy) which looks compelling in the final presentation. And the strict no-number constraint on the LLM is philosophically sound — it forces the model to reason about causality (what will be affected) rather than curve-fitting price history.
Want me to write the actual Python code for any of these phases — dataset builder, env class, or reward function?
give me a well crafted prompt that i should give to my coding agent for creating this project. give prompts in chuck for part by part so that agent doesn't start to hallucinate due to too much context. and also write what is that part supposed to build and how to check
18:13
Architected incremental prompts to prevent agent hallucination
Architected incremental prompts to prevent agent hallucination
Coding agent prompt guide — chunked by part
Each chunk below has: what it builds, the prompt to give, and how to verify.

Part 1 — Project scaffold & config
What it builds: Folder structure, config.py with all constants (API keys, keyword lists, trusted domains, asset tickers), and requirements.txt. Nothing runs yet — just the skeleton.
Prompt:
Create a Python project scaffold for a news-driven stock prediction dataset builder.

Folder structure:
  news_stock_env/
    config.py
    requirements.txt
    data/           (empty, for output CSVs)
    logs/           (empty)

In config.py define:
- NEWS_API_KEY = "" (placeholder)
- OPENAI_API_KEY = "" (placeholder)
- TRUSTED_DOMAINS = ["bbc.co.uk", "cnbc.com", "reuters.com", "bloomberg.com", 
    "ft.com", "wsj.com", "apnews.com", "theguardian.com", "forbes.com"]
- ASSET_KEYWORDS = ["gold", "silver", "oil", "crude", "rare earth", "copper", 
    "natural gas", "lithium"]
- MACRO_KEYWORDS = ["war", "sanctions", "supply chain", "rate hike", "inflation", 
    "recession", "federal reserve", "OPEC", "interest rate", "GDP", "unemployment",
    "trade war", "tariff", "scarcity", "demand shock", "geopolitical"]
- ALL_KEYWORDS = ASSET_KEYWORDS + MACRO_KEYWORDS
- FIXED_ASSETS = {"gold": "GLD", "silver": "SLV", "oil": "USO"}
- SP500_TICKERS (a hardcoded list of 50 large-cap tickers like AAPL, MSFT, NVDA, 
    GOOGL, META, AMZN, TSLA, XOM, CVX, JPM, BAC, etc.)
- SHORT_TERM_ARTICLES_PER_DATE = 10
- LONG_TERM_DAYS_BACK = 7
- TOP_N_MOVERS = 5

In requirements.txt include: newsapi-python, yfinance, openai, pandas, 
  python-dotenv, tqdm, requests

No main logic yet. Just config and folder setup.
How to verify:
Run python -c "from news_stock_env.config import ALL_KEYWORDS; print(len(ALL_KEYWORDS))" — should print 26+
All folders exist, requirements.txt has all packages

Part 2 — yfinance module (ground truth fetcher)
What it builds: A standalone stock_fetcher.py that, given a date, returns top 5 gainers, top 5 losers from SP500_TICKERS, and direction (UP/DOWN/NEUTRAL) for the 3 fixed assets.
Prompt:
Create news_stock_env/stock_fetcher.py

It must expose one public function:
  get_stock_predictions(date_str: str) -> dict

Where date_str is "YYYY-MM-DD" and the return value is:
{
  "gainers": ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5"],  # ranked best to worst
  "losers":  ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5"],  # ranked worst to best
  "assets":  {"gold": "UP", "silver": "DOWN", "oil": "NEUTRAL"}
}

Implementation rules:
- Use yfinance. For each ticker in SP500_TICKERS from config.py, download 
  1-day OHLCV data for the given date using yf.download(ticker, start=date_str, 
  end=next_day_str, progress=False).
- Compute pct_change = (close - open) / open * 100 for that single day.
- Handle missing data gracefully (skip tickers where download fails or returns empty).
- Sort all tickers by pct_change descending. Top 5 = gainers, bottom 5 = losers.
- For FIXED_ASSETS (GLD, SLV, USO), compute their pct_change separately:
    UP if pct_change > 0.3
    DOWN if pct_change < -0.3
    NEUTRAL otherwise
- Add a small CLI test at the bottom under if __name__ == "__main__": that calls
  get_stock_predictions("2024-01-15") and pretty-prints the result.
- Import SP500_TICKERS, FIXED_ASSETS from config.py (use relative import).
How to verify:
cd news_stock_env && python stock_fetcher.py
Output should have exactly 5 gainers, 5 losers, and 3 asset directions
Try 2–3 different dates and check the tickers look plausible (not all the same)

Part 3 — NewsAPI module
What it builds: news_fetcher.py with two functions — one for long-term context (7-day window, top 10 by relevancy) and one for short-term context (single day, top 10 by popularity). Returns clean article objects.
Prompt:
Create news_stock_env/news_fetcher.py

Expose two public functions:

1. get_long_term_context(date_str: str) -> list[dict]
   - date range: (date_str - 7 days) to date_str
   - sortBy: "relevancy"
   - pageSize: 10
   - Returns list of 10 dicts: {"title": ..., "description": ..., "source": ..., "url": ...}

2. get_short_term_articles(date_str: str) -> list[dict]
   - date range: date_str to date_str (single day)
   - sortBy: "popularity"
   - pageSize: 10
   - Returns list of 10 dicts: same schema as above

Both functions must:
- Use the newsapi-python client: from newsapi import NewsApiClient
- Use NEWS_API_KEY from config.py
- Use q = " OR ".join(ALL_KEYWORDS[:15]) as the query string (first 15 keywords)
- Use sources param built from TRUSTED_DOMAINS (join with comma)
- Strip None values from title/description, replacing with ""
- Catch all API exceptions and return empty list with a logged warning

Add a CLI test under if __name__ == "__main__": that fetches both for "2024-01-15" 
and prints the count and first article title of each.
How to verify:
python news_fetcher.py with a valid NEWS_API_KEY in config
Should print "Long-term: 10 articles" and "Short-term: X articles"
If API key is missing, should fail gracefully with a warning, not a crash

Part 4 — Difficulty labeller
What it builds: difficulty_labeller.py — a GPT-4o-mini call that takes a single article and the day's movers list, and returns "Easy", "Medium", or "Hard".
Prompt:
Create news_stock_env/difficulty_labeller.py

Expose one public function:
  label_difficulty(article: dict, stock_predictions: dict) -> str
  Returns exactly one of: "Easy", "Medium", "Hard"

The function should call GPT-4o-mini (model="gpt-4o-mini") with this exact system prompt:
  "You are a financial analyst assistant. Classify news articles into difficulty levels 
   for a stock prediction task. Respond with exactly one word: Easy, Medium, or Hard.
   
   Easy: article directly names a specific stock or commodity AND its price direction 
         (e.g. 'Oil surges 3%', 'Apple stock falls after earnings').
   Medium: article references a sector, commodity, or company but does NOT state 
           the price direction (e.g. 'Oil reserves reported lower than expected').
   Hard: article is about a macro event with no direct reference to specific stocks 
         or commodities (e.g. 'US imposes new tariffs on China')."

User message should include:
  - Article title and description
  - The tickers in stock_predictions["gainers"] + stock_predictions["losers"]
  - Ask: "Classify this article's difficulty level."

Parse the response: strip whitespace, extract first word, validate it's in 
["Easy","Medium","Hard"]. If not, default to "Medium".

Add a simple retry (up to 2 retries) on openai API errors.

Add a CLI test under if __name__ == "__main__".
How to verify:
Feed it a clearly easy article manually (e.g. title = "Gold hits record high as investors flee stocks") — should return "Easy"
Feed it a hard one (e.g. "North Korea conducts missile test") — should return "Hard"
Test that an unexpected GPT response defaults to "Medium" gracefully

Part 5 — Dataset builder (main pipeline)
What it builds: build_dataset.py — orchestrates parts 2, 3, 4 to generate the full CSV. Loops over a date range, creates 10 rows per date (one per short-term article), saves incrementally.
Prompt:
Create news_stock_env/build_dataset.py

This is the main orchestration script. It must:

1. Accept CLI arguments: 
   --start_date YYYY-MM-DD
   --end_date YYYY-MM-DD
   --output data/dataset.csv

2. For each date in the range (business days only, skip weekends):
   a. Call get_stock_predictions(date_str) from stock_fetcher.py
   b. Call get_long_term_context(date_str) from news_fetcher.py → one shared 
      string per date (join all 10 article titles+descriptions with " | ")
   c. Call get_short_term_articles(date_str) from news_fetcher.py → list of 10
   d. For each of the 10 short-term articles (i = 0..9):
      - Call label_difficulty(article, stock_predictions)
      - Create one dataset row:
          ID: auto-increment integer
          date: date_str
          long_term_context: the joined string from step b
          short_term_context: this single article's title + " — " + description
          stock_predictions: json.dumps(stock_predictions)
          difficulty: the label
   e. Append all 10 rows to the output CSV immediately after each date 
      (so progress is saved even if script crashes mid-run)

3. Use tqdm for a progress bar over dates.
4. Skip a date entirely (with a warning log) if stock_fetcher returns empty gainers.
5. Add a --resume flag: if the output CSV already exists, skip dates already present.
6. At the end, print a summary: total rows, Easy/Medium/Hard counts.

Use pandas for CSV writing. Use logging (not print) for all status messages.
How to verify:
Run with a 5-day range: python build_dataset.py --start_date 2024-01-15 --end_date 2024-01-19 --output data/test.csv
Open test.csv — should have ~50 rows (10 per business day)
Check columns: all 6 present, no empty stock_predictions, difficulty only has 3 valid values
Kill it midway, re-run with --resume — should not duplicate rows

Part 6 — OpenEnv environment class
What it builds: env.py — the RL environment that wraps the dataset. Implements reset() and step() per OpenEnv spec.
Prompt:
Create news_stock_env/env.py

Implement a class NewsStockEnv following the OpenEnv gym-style interface.

__init__(self, dataset_path: str, shuffle: bool = True):
  - Load the CSV from dataset_path into a pandas DataFrame
  - Optionally shuffle rows (for training)
  - Keep an internal index pointer self._idx = 0

reset(self, idx: int = None) -> str:
  - If idx is None, use self._idx and increment it (wraps around)
  - Load the row at that index
  - Store self._ground_truth = json.loads(row["stock_predictions"])
  - Store self._difficulty = row["difficulty"]
  - Return this formatted state string (exactly this format):
    """
    === MARKET CONTEXT ===
    [LONG-TERM — past 7 days]
    {long_term_context}

    [TODAY'S NEWS]
    {short_term_context}

    === YOUR TASK ===
    Predict which assets will be affected today.
    Output a JSON object with keys:
      "gainers": list of up to 5 stock tickers, ranked most to least gain
      "losers": list of up to 5 stock tickers, ranked most to least loss
      "assets": dict with keys "gold", "silver", "oil" and values "UP", "DOWN", or "NEUTRAL"
    Output JSON only. No explanation. No numbers.
    """

step(self, action: str) -> tuple[float, dict]:
  - Parse action as JSON. If parsing fails, return (0.0, {"error": "invalid JSON"})
  - Call self._compute_reward(action_dict, self._ground_truth)
  - Apply difficulty multiplier: Easy=1.0, Medium=1.5, Hard=2.0
  - Return (reward * multiplier, {"ground_truth": self._ground_truth, 
                                   "difficulty": self._difficulty,
                                   "raw_reward": reward})

_compute_reward(self, prediction: dict, ground_truth: dict) -> float:
  - Gainers (ranked): +2 if ticker in correct position, +1 if in list wrong position
  - Losers (ranked): same scoring
  - Assets: +1 for each correct direction (gold/silver/oil)
  - Return total float score

Add a __len__ method returning dataset size.
Add a CLI smoke test under if __name__ == "__main__" that loads data/test.csv,
calls reset(), prints the state, then calls step() with a dummy JSON prediction.
How to verify:
python env.py should print a formatted state string and a reward value
Call step('{"gainers":[],"losers":[],"assets":{}}') — reward should be 0.0
Call step('not json') — should return (0.0, {"error": "invalid JSON"}) without crashing

Part 7 — Training loop (OpenEnv integration)
What it builds: train.py — connects the env to an LLM (via OpenAI or local Llama) using the GRPO-style loop from OpenEnv's tutorial.
Prompt:
Create news_stock_env/train.py

This script runs the RL training loop using the NewsStockEnv.

It should:
1. Load the env: env = NewsStockEnv("data/dataset.csv", shuffle=True)
2. Define a rollout function:
   def rollout(state: str) -> tuple[str, float]:
     - Call OpenAI chat completion with model="gpt-4o-mini" (or swap for Llama)
     - System prompt: "You are a financial analyst. Output ONLY valid JSON. 
       Never output prices, numbers, or explanations."
     - User message: the state string from env.reset()
     - Parse the response and call env.step(response_text)
     - Return (response_text, reward)

3. Run N_EPISODES = 100 episodes:
   - For each episode:
     a. state = env.reset()
     b. action, reward = rollout(state)
     c. Log to logs/training_log.csv: episode, reward, difficulty, action (truncated)
     d. Print running average reward every 10 episodes

4. At the end, print:
   - Average reward by difficulty level (Easy / Medium / Hard)
   - Top 3 highest-reward episodes

Keep the LLM call in a try/except — skip the episode and log reward=0 on failure.

This script is intentionally simple (no backprop yet) — it's a zero-shot 
evaluation baseline to verify the reward function works end-to-end before 
plugging into full GRPO training per the OpenEnv tutorial.
How to verify:
python train.py should run 100 episodes and save logs/training_log.csv
Check the log: rewards should vary (not all 0), difficulty column should have all 3 values
Average Hard reward should be lower than Easy reward — if not, the labeller may be miscategorising

Final integration check (give this after all parts are done)
Prompt:
Do a full end-to-end integration check of the project:

1. Run: python build_dataset.py --start_date 2024-01-15 --end_date 2024-01-26 --output data/full_test.csv
   Confirm: ~100 rows generated, all 6 columns populated, no NaN in stock_predictions.

2. Run: python env.py
   Confirm: state string prints correctly, reward function runs without error.

3. Run: python train.py
   Confirm: 100 episodes complete, training_log.csv saved, reward breakdown by difficulty prints.

4. Print the distribution of difficulty labels in data/full_test.csv using pandas value_counts().
   Expected: roughly 30% Easy, 40% Medium, 30% Hard.

5. Check data/full_test.csv for any rows where stock_predictions has fewer than 5 gainers 
   or losers — log those dates as potentially bad yfinance days (holidays, market closures).

Report any errors found and fix them.

This sequence gives you a working project in 7 clean steps, with each agent call isolated enough that context doesn't bleed. Parts 2 and 3 can be given in parallel if your agent supports it.
we dont want to feed each article of each day one by one, we have to predict the result for end of the day, hene we will give all the article of the day  in one go in short term memory. 
give me list of pros and cons for doing this way, if we do this way then how will be differentiate between tasks being easy medium and hard?
19:46
Evaluated consolidated dataset approach and difficulty metrics
Evaluated consolidated dataset approach and difficulty metrics
Bundled daily context — pros & cons
Pros
Realistic framing. A real analyst reads everything available before market close, not one article in isolation. The model learns a more honest task.
Richer signal per prediction. Multiple articles can corroborate or contradict each other — the model can reason about consensus vs noise, which is a genuinely harder and more valuable skill.
Cleaner dataset. 1 row per date instead of 10 means no duplicate stock_predictions ground truth spread across 10 rows. Much easier to audit.
Better for RL episodes. One episode = one full trading day. The reward is unambiguous — no confusion about whether the model is predicting "today" from article 3 vs article 7.
Simpler training loop. No need to track which article index you're on within a date.
Cons
Much larger context per prompt. 10 articles bundled together can easily hit 3000–5000 tokens per state. With smaller/local Llama models this becomes a problem.
10x fewer training examples. If you were building over 100 dates, you had 1000 rows before. Now you have 100. Dataset size shrinks significantly.
Harder to isolate what the model learned. If it gets a Hard day wrong, you can't tell which article confused it — the signal is noisier for debugging.
Redundant articles dilute the signal. Some of the 10 articles on a given day may be about completely unrelated topics (sports, politics with no market angle). They add noise without adding value.

Rethinking difficulty with bundled articles
Since difficulty can no longer be about a single article, it needs to become a day-level property. Here are three clean approaches you can combine:

Approach 1 — Signal density score (best one)
Count how many of the 10 articles have a direct reference to the assets that actually moved that day.
signal_articles = number of articles that mention a top-5 gainer/loser or fixed asset

Easy   → signal_articles >= 6   (majority of articles point directly to movers)
Medium → signal_articles 3–5    (some signal, some noise)
Hard   → signal_articles <= 2   (movers are barely mentioned — model must infer)
This is the most principled version because difficulty is now directly tied to how much the context justifies the answer.

Approach 2 — Causal distance (semantic, GPT-labelled)
Give GPT all 10 headlines + the actual movers list and ask it to classify the reasoning chain required:
Easy   → movers are named AND their direction is stated in the articles
Medium → movers are named but direction must be inferred from context
Hard   → movers are not named; model must connect a macro event to market impact
         (e.g. "Fed raises rates" → banks move, but articles don't say so)
This is the most meaningful for training because it captures the inference gap — which is exactly what you want the model to learn to close.

Approach 3 — Outcome surprise score (quantitative, no GPT needed)
Use the articles' publication times and pre-market consensus as a proxy:
Easy   → the day's top movers were already trending in the news (high article overlap)
Medium → partial overlap between news topics and movers
Hard   → movers that day had zero news coverage → pure macro/sentiment driven move
You can approximate this by checking if any of the top 5 gainer/loser tickers appear in any article text. Zero appearances = Hard.

Recommended combination
Use Approach 1 for automated labelling (fast, no GPT cost, fully deterministic) and Approach 2 as a GPT verification pass on a sample of 20% of rows to validate the labels are meaningful. Approach 3 can be a secondary feature column rather than the primary difficulty signal.
The revised dataset row becomes:
ID | Date | long_term_context | short_term_context (all 10 articles) 
   | stock_predictions | signal_article_count | difficulty
signal_article_count is a useful raw feature to keep — it gives the model (and you) a quantitative handle on why a day was Hard, not just a label.


