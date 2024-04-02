# IMDb Movie Scraper and Sentiment Analysis

## Project Status: Work in Progress

### Overview

The IMDb Movie Scraper and Sentiment Analysis project aims to automate the process of gathering movie data from IMDb, categorizing user reviews based on sentiment analysis, and summarizing the findings into a dashboard. This project is being developed to provide users with valuable insights into movie reviews and ratings.

### Features

- **Movie Title Input**: Users can input the title of the movie they want to analyze.
- **Data Scraping**: Automatically scrapes data (ratings, reviews, storyline, etc.) from IMDb using BeautifulSoup.
- **Sentiment Analysis**: Reviews are categorized as positive or negative based on sentiment analysis.
- **Summarization Task**: Summarizes positive and negative reviews for each movie.
- **Dashboard Display**: Displays the extracted information in a dashboard for easy understanding.

### Technologies Used

- **FLAN T5**: Utilized for the Language Model.
- **PEFT LoRA**: Used for fine-tuning the LLM for sentiment analysis.

### Completed Progress

- Successfully fine tuning FLAN T5 for sentiment analysis tasks with results of 0.94 accuracy and 0.237 test loss.
- Successfully scraped important information from the IMDb website using beautiful soup.
  
### Current Progress

- Working on improving summarization task fine tuning result.
- Continuous testing and refinement ongoing.

### Future Plans

- Enhance summarization algorithm for more accurate results.
- Developing dashboards.
- Automation.
