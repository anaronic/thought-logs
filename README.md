# ThoughtLogs

ThoughtLogs is a modern, full-stack journaling and self-optimization application. It features a FastAPI backend and a responsive frontend built with HTML, CSS, and JavaScript. The app helps users log thoughts, analyze cognitive load, optimize learning pathways, and predict future trends, all in a beautiful, calming interface.

## Features
- **Thought Logging:** Add, edit, delete, and search thoughts with categories, priorities, and attachments.
- **Future Self Predictor:** AI-driven predictions and advice based on your thought patterns.
- **Cognitive Load Analysis:** Visualize and manage your mental workload.
- **Learning Pathway Optimizer:** Personalized skill recommendations and learning paths.
- **Pomodoro Timer:** Built-in productivity timer.
- **Live Search & Filtering:** Instantly filter thoughts by text, category, and urgency.
- **Modern UI/UX:** Responsive design, dark mode, animated backgrounds, and ambient particles.

## Tech Stack
- **Backend:** FastAPI, Motor (MongoDB async driver), MongoDB Atlas
- **Frontend:** HTML5, CSS3 (Grid/Flexbox, custom animations), JavaScript (vanilla)
- **Other:** Uvicorn (ASGI server)

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js (optional, for advanced frontend tooling)
- MongoDB Atlas account (or local MongoDB instance)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/thoughtlogs.git
   cd thoughtlogs
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure MongoDB:**
   - Update the `MONGO_DETAILS` string in `main.py` with your MongoDB Atlas URI or local MongoDB URI.
4. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```
5. **Access the app:**
   - Open your browser at [http://localhost:8000](http://localhost:8000)

## Usage
- Log new thoughts, assign categories and priorities, and attach files.
- Use the widgets for insights, learning recommendations, and productivity.
- All features are accessible from the main dashboard.

## API Endpoints
- `GET /` — Main frontend (HTML)
- `POST /thoughts/` — Add a new thought
- `GET /thoughts/` — List/search thoughts (supports `search`, `category`, `priority` query params)
- `PUT /thoughts/{id}` — Update a thought
- `DELETE /thoughts/{id}` — Delete a thought
- `POST /thoughts/{id}/attachments/` — Upload attachments
- `GET /categories/` — List categories
- `POST /categories/` — Add a category
- `PUT /categories/{old_name}` — Rename a category
- `DELETE /categories/{name}` — Delete a category
- `GET /cognitive-load/` — Get cognitive load analysis
- `GET /learning-pathway/` — Get learning pathway insights
- `GET /future-self/` — Get future self prediction

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License
[MIT](LICENSE)

## Contact
- **Author:** Anaranyo Sarkar
- **Email:** anaranyo2705@gmail.com
- **GitHub:** [yourusername](https://github.com/yourusername)

---

*ThoughtLogs — Reflect. Analyze. Grow.* 