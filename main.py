from typing import Union, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uuid
import motor.motor_asyncio
from datetime import datetime, timedelta
import os
import re
from collections import Counter

app = FastAPI()

# MongoDB Atlas connection
MONGO_DETAILS = "mongodb+srv://anaranyo2705:La7wgC48S4Qdw5gf@cluster0.cs8qrpa.mongodb.net/"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
db = client.thoughtlogger
thoughts_collection = db.thoughts
categories_collection = db.categories

# Ensure uploads directory exists
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Thought model
class Thought(BaseModel):
    id: Optional[str] = None
    content: str
    category: Optional[str] = None
    attachments: List[str] = []
    priority: Optional[str] = None  # e.g., 'urgent', 'normal', 'low', or 1-5
    timestamp: Union[str, None] = None

# Category model
class Category(BaseModel):
    name: str

# Cognitive Load Analysis
class CognitiveLoadInsight(BaseModel):
    score: float  # 0-100, where 0 = low load, 100 = high load
    level: str  # "low", "medium", "high"
    indicators: List[str]
    suggestions: List[str]
    patterns: dict

# Learning Pathway Optimizer
class LearningPathwayInsight(BaseModel):
    current_level: str  # "beginner", "intermediate", "advanced"
    recommended_skills: List[str]
    learning_path: List[dict]
    time_estimates: dict
    confidence_score: float  # 0-100
    next_steps: List[str]

# Future Self Predictor
class FutureSelfPrediction(BaseModel):
    summary: str
    predicted_trends: List[str]
    risk_factors: List[str]
    positive_outcomes: List[str]
    advice: List[str]
    time_horizon: str  # e.g., '1-3 months', '6-12 months'

def analyze_cognitive_load(thoughts: List[dict]) -> CognitiveLoadInsight:
    if not thoughts:
        return CognitiveLoadInsight(
            score=0,
            level="low",
            indicators=["No thoughts logged yet"],
            suggestions=["Start logging thoughts to get personalized insights", "Try logging 3-5 thoughts to see patterns emerge"],
            patterns={}
        )
    
    # Analyze recent thoughts (last 10 or last 24 hours)
    recent_thoughts = thoughts[-10:] if len(thoughts) > 10 else thoughts
    
    indicators = []
    suggestions = []
    patterns = {}
    
    # 1. Priority Analysis
    urgent_count = sum(1 for t in recent_thoughts if t.get('priority') == 'urgent')
    if urgent_count > 0:
        urgent_ratio = urgent_count / len(recent_thoughts)
        if urgent_ratio > 0.5:
            indicators.append(f"High urgency thoughts ({urgent_ratio:.0%})")
            suggestions.append("Consider breaking down urgent tasks")
        patterns['urgent_ratio'] = urgent_ratio
    
    # 2. Category Switching (topic switching)
    categories = [t.get('category') for t in recent_thoughts if t.get('category')]
    if len(categories) > 1:
        unique_categories = len(set(categories))
        if unique_categories > len(categories) * 0.7:
            indicators.append("Frequent topic switching")
            suggestions.append("Try focusing on one topic at a time")
        patterns['category_switching'] = unique_categories / len(categories)
    
    # 3. Temporal Patterns
    if len(thoughts) >= 2:
        # Check for clustering (many thoughts in short time)
        timestamps = [datetime.strptime(t['timestamp'], "%B %d, %Y, %I:%M %p") for t in thoughts if t.get('timestamp')]
        if len(timestamps) >= 2:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
            recent_diffs = time_diffs[-5:] if len(time_diffs) > 5 else time_diffs
            avg_time_diff = sum(recent_diffs) / len(recent_diffs)
            
            if avg_time_diff < 1:  # Less than 1 hour between thoughts
                indicators.append("Thought clustering (many thoughts in short time)")
                suggestions.append("Consider taking breaks between thinking sessions")
            patterns['avg_time_between_thoughts'] = avg_time_diff
            
            # Time of day analysis
            hours = [ts.hour for ts in timestamps]
            if len(hours) >= 3:
                # Check for late night thinking
                late_night_count = sum(1 for h in hours if h >= 22 or h <= 4)
                if late_night_count > len(hours) * 0.4:
                    indicators.append("Late night thinking patterns")
                    suggestions.append("Consider earlier thinking sessions for better clarity")
                patterns['late_night_ratio'] = late_night_count / len(hours)
    
    # 4. Content Analysis (enhanced linguistic complexity)
    total_words = sum(len(t.get('content', '').split()) for t in recent_thoughts)
    avg_words = total_words / len(recent_thoughts)
    
    # Enhanced complexity scoring
    complexity_score = 0
    for thought in recent_thoughts:
        content = thought.get('content', '').lower()
        # Sentence complexity (periods, commas, semicolons)
        sentence_markers = content.count('.') + content.count(',') + content.count(';')
        # Question complexity
        question_markers = content.count('?')
        # Emotional intensity words
        intensity_words = ['very', 'extremely', 'really', 'absolutely', 'completely', 'totally']
        intensity_count = sum(1 for word in intensity_words if word in content)
        # Technical terms (simple heuristic)
        technical_indicators = ['because', 'therefore', 'however', 'although', 'nevertheless']
        technical_count = sum(1 for word in technical_indicators if word in content)
        
        complexity_score += sentence_markers + question_markers * 2 + intensity_count + technical_count
    
    avg_complexity = complexity_score / len(recent_thoughts)
    
    if avg_words > 50:
        indicators.append("Long, complex thoughts")
        suggestions.append("Try breaking complex thoughts into smaller pieces")
    elif avg_words < 10:
        indicators.append("Very brief thoughts")
        suggestions.append("Consider expanding on your thoughts for better clarity")
    
    if avg_complexity > 8:
        indicators.append("High cognitive complexity in thoughts")
        suggestions.append("Consider simplifying your thinking process")
    
    patterns['avg_words_per_thought'] = avg_words
    patterns['avg_complexity_score'] = avg_complexity
    
    # 5. Emotional/Stress Indicators (enhanced keyword analysis)
    stress_words = ['stress', 'overwhelmed', 'anxious', 'worried', 'tired', 'exhausted', 'burnout', 'pressure', 'frustrated', 'confused']
    positive_words = ['excited', 'happy', 'great', 'wonderful', 'amazing', 'progress', 'success', 'achieved']
    
    stress_count = sum(1 for t in recent_thoughts 
                      for word in stress_words 
                      if word in t.get('content', '').lower())
    positive_count = sum(1 for t in recent_thoughts 
                        for word in positive_words 
                        if word in t.get('content', '').lower())
    
    if stress_count > 0:
        stress_ratio = stress_count / len(recent_thoughts)
        if stress_ratio > 0.3:
            indicators.append("Stress-related language detected")
            suggestions.append("Consider stress management techniques or breaks")
        patterns['stress_indicators'] = stress_ratio
    
    if positive_count > 0:
        positive_ratio = positive_count / len(recent_thoughts)
        patterns['positive_indicators'] = positive_ratio
    
    # 6. Thought Frequency Analysis (for low data scenarios)
    if len(thoughts) >= 3:
        # Check if user is logging thoughts regularly
        days_with_thoughts = len(set(t['timestamp'].split(',')[0] for t in thoughts if t.get('timestamp')))
        total_days = (datetime.now() - datetime.strptime(thoughts[0]['timestamp'], "%B %d, %Y, %I:%M %p")).days + 1
        
        if total_days > 0:
            consistency_ratio = days_with_thoughts / total_days
            if consistency_ratio < 0.3:
                indicators.append("Irregular thought logging")
                suggestions.append("Try logging thoughts daily for better insights")
            patterns['logging_consistency'] = consistency_ratio
    
    # 7. Smart defaults for new users (less than 5 thoughts)
    if len(thoughts) < 5:
        suggestions.extend([
            "You're just getting started! Log a few more thoughts to see personalized patterns",
            "Try logging thoughts at different times of day",
            "Experiment with different categories and priorities"
        ])
    
    # Calculate overall score with enhanced weighting
    score = 0
    if indicators:
        # Enhanced scoring system
        base_score = len(indicators) * 12
        
        # Weight different indicators
        if any('urgent' in ind.lower() for ind in indicators):
            base_score += 25
        if any('stress' in ind.lower() for ind in indicators):
            base_score += 30
        if any('clustering' in ind.lower() for ind in indicators):
            base_score += 20
        if any('complexity' in ind.lower() for ind in indicators):
            base_score += 15
        if any('late night' in ind.lower() for ind in indicators):
            base_score += 18
        
        score = min(100, base_score)
    
    # Determine level with more nuanced thresholds
    if score < 25:
        level = "low"
        if not suggestions:
            suggestions.append("Your cognitive load appears manageable")
    elif score < 60:
        level = "medium"
        if not suggestions:
            suggestions.append("Your cognitive load is moderate - consider taking short breaks")
    else:
        level = "high"
        suggestions.append("Consider taking a longer break or reducing workload")
    
    return CognitiveLoadInsight(
        score=score,
        level=level,
        indicators=indicators,
        suggestions=suggestions,
        patterns=patterns
    )

# Helper
def thought_helper(thought) -> dict:
    return {
        "id": thought["id"],
        "content": thought["content"],
        "category": thought.get("category"),
        "attachments": thought.get("attachments", []),
        "priority": thought.get("priority"),
        "timestamp": thought.get("timestamp"),
    }

def category_helper(category) -> dict:
    return {"name": category["name"]}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("Templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/cognitive-load/", response_model=CognitiveLoadInsight)
async def get_cognitive_load():
    """Analyze cognitive load based on recent thoughts"""
    thoughts = []
    async for thought in thoughts_collection.find():
        thoughts.append(thought_helper(thought))
    
    return analyze_cognitive_load(thoughts)

@app.get("/learning-pathway/", response_model=LearningPathwayInsight)
async def get_learning_pathway():
    """Analyze learning pathway and provide skill development recommendations"""
    thoughts = []
    async for thought in thoughts_collection.find():
        thoughts.append(thought_helper(thought))
    
    return analyze_learning_pathway(thoughts)

@app.post("/thoughts/", response_model=Thought)
async def create_thought(thought: Thought):
    thought.id = str(uuid.uuid4())
    thought.timestamp = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    thought_dict = thought.dict()
    await thoughts_collection.insert_one(thought_dict)
    return thought

@app.get("/thoughts/", response_model=List[Thought])
async def get_thoughts(search: Optional[str] = None, category: Optional[str] = None, priority: Optional[str] = None):
    query = {}
    if search:
        query["content"] = {"$regex": search, "$options": "i"}
    if category:
        query["category"] = category
    if priority:
        query["priority"] = priority
    thoughts = []
    async for thought in thoughts_collection.find(query):
        thoughts.append(thought_helper(thought))
    return thoughts

@app.get("/thoughts/{thought_id}", response_model=Thought)
async def get_thought(thought_id: str):
    thought = await thoughts_collection.find_one({"id": thought_id})
    if thought:
        return thought_helper(thought)
    raise HTTPException(status_code=404, detail="Thought not found")

@app.put("/thoughts/{thought_id}", response_model=Thought)
async def update_thought(thought_id: str, thought: Thought):
    update_data = thought.dict()
    update_data["id"] = thought_id
    result = await thoughts_collection.replace_one({"id": thought_id}, update_data)
    if result.modified_count == 1:
        return update_data
    raise HTTPException(status_code=404, detail="Thought not found")

@app.delete("/thoughts/{thought_id}")
async def delete_thought(thought_id: str):
    result = await thoughts_collection.delete_one({"id": thought_id})
    if result.deleted_count == 1:
        return {"message": "Thought deleted"}
    raise HTTPException(status_code=404, detail="Thought not found")

# Category endpoints
@app.get("/categories/", response_model=List[Category])
async def list_categories():
    categories = []
    async for category in categories_collection.find():
        categories.append(category_helper(category))
    return categories

@app.post("/categories/", response_model=Category)
async def add_category(category: Category):
    if await categories_collection.find_one({"name": category.name}):
        raise HTTPException(status_code=400, detail="Category already exists")
    await categories_collection.insert_one(category.dict())
    return category

@app.put("/categories/{old_name}", response_model=Category)
async def rename_category(old_name: str, new_name: str = Form(...)):
    result = await categories_collection.update_one({"name": old_name}, {"$set": {"name": new_name}})
    if result.modified_count == 1:
        return {"name": new_name}
    raise HTTPException(status_code=404, detail="Category not found")

@app.delete("/categories/{name}")
async def delete_category(name: str):
    result = await categories_collection.delete_one({"name": name})
    if result.deleted_count == 1:
        return {"message": "Category deleted"}
    raise HTTPException(status_code=404, detail="Category not found")

# Attachment upload endpoint
@app.post("/thoughts/{thought_id}/attachments/")
async def upload_attachment(thought_id: str, file: UploadFile = File(...)):
    filename = file.filename or f"attachment_{uuid.uuid4()}"
    file_location = os.path.join(UPLOAD_DIR, filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    # Update thought with attachment URL
    url = f"/static/uploads/{filename}"
    result = await thoughts_collection.update_one({"id": thought_id}, {"$push": {"attachments": url}})
    if result.modified_count == 1:
        return {"url": url}
    raise HTTPException(status_code=404, detail="Thought not found")

def analyze_learning_pathway(thoughts: List[dict]) -> LearningPathwayInsight:
    if not thoughts:
        return LearningPathwayInsight(
            current_level="beginner",
            recommended_skills=["Start logging thoughts to discover your learning patterns"],
            learning_path=[],
            time_estimates={},
            confidence_score=0,
            next_steps=["Begin thought logging", "Add learning-related thoughts"]
        )
    
    # Analyze recent thoughts (last 20 or all if less)
    recent_thoughts = thoughts[-20:] if len(thoughts) > 20 else thoughts
    
    # 1. Skill Detection from Content
    skill_keywords = {
        "programming": ["code", "programming", "python", "javascript", "react", "api", "database", "algorithm", "function", "class", "debug", "git", "deployment"],
        "data_science": ["data", "analysis", "machine learning", "ml", "ai", "statistics", "visualization", "pandas", "numpy", "tensorflow", "model", "prediction"],
        "design": ["design", "ui", "ux", "interface", "layout", "color", "typography", "wireframe", "prototype", "user experience", "visual"],
        "business": ["business", "strategy", "marketing", "sales", "finance", "management", "leadership", "startup", "entrepreneur", "market", "customer"],
        "writing": ["write", "content", "blog", "article", "copy", "story", "narrative", "communication", "editing", "publishing"],
        "productivity": ["productivity", "efficiency", "workflow", "automation", "tools", "process", "optimization", "time management", "organization"],
        "creativity": ["creative", "art", "music", "design", "innovation", "ideas", "brainstorming", "inspiration", "artistic", "expression"],
        "communication": ["communication", "presentation", "speaking", "writing", "meeting", "collaboration", "teamwork", "leadership", "negotiation"],
        "problem_solving": ["problem", "solve", "solution", "troubleshoot", "debug", "fix", "resolve", "issue", "challenge", "analysis"],
        "learning": ["learn", "study", "course", "tutorial", "education", "knowledge", "skill", "practice", "improve", "master"]
    }
    
    skill_counts = {skill: 0 for skill in skill_keywords.keys()}
    skill_mentions = {skill: [] for skill in skill_keywords.keys()}
    
    for thought in recent_thoughts:
        content = thought.get('content', '').lower()
        for skill, keywords in skill_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    skill_counts[skill] += 1
                    skill_mentions[skill].append(thought.get('content', '')[:100] + "...")
                    break
    
    # 2. Determine Current Level
    total_skill_mentions = sum(skill_counts.values())
    if total_skill_mentions == 0:
        current_level = "beginner"
    elif total_skill_mentions < 5:
        current_level = "beginner"
    elif total_skill_mentions < 15:
        current_level = "intermediate"
    else:
        current_level = "advanced"
    
    # 3. Identify Top Skills and Gaps
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_skill_names = [skill for skill, count in top_skills if count > 0]
    
    # Find complementary skills
    skill_relationships = {
        "programming": ["data_science", "problem_solving", "productivity"],
        "data_science": ["programming", "problem_solving", "communication"],
        "design": ["creativity", "communication", "productivity"],
        "business": ["communication", "problem_solving", "productivity"],
        "writing": ["communication", "creativity", "learning"],
        "productivity": ["problem_solving", "learning", "communication"],
        "creativity": ["design", "writing", "learning"],
        "communication": ["business", "writing", "productivity"],
        "problem_solving": ["programming", "data_science", "productivity"],
        "learning": ["productivity", "communication", "creativity"]
    }
    
    recommended_skills = []
    for skill, count in top_skills:
        if count > 0 and skill in skill_relationships:
            for related_skill in skill_relationships[skill]:
                if related_skill in skill_counts and skill_counts[related_skill] == 0:  # Skill not mentioned
                    recommended_skills.append(related_skill)
    
    # Add general recommendations if no specific skills found
    if not recommended_skills:
        recommended_skills = ["learning", "productivity", "communication"]
    # Deduplicate recommended_skills while preserving order
    seen = set()
    recommended_skills = [x for x in recommended_skills if not (x in seen or seen.add(x))]
    
    # 4. Create Learning Path
    learning_path = []
    for skill in recommended_skills[:3]:  # Top 3 recommendations
        skill_name = skill.replace('_', ' ').title()
        path_item = {
            "skill": skill_name,
            "current_level": "beginner",
            "target_level": "intermediate",
            "resources": get_learning_resources(skill),
            "estimated_hours": get_time_estimate(skill, "beginner", "intermediate"),
            "priority": "high" if skill in top_skill_names else "medium"
        }
        learning_path.append(path_item)
    
    # 5. Time Estimates
    time_estimates = {
        "beginner_to_intermediate": 40,
        "intermediate_to_advanced": 80,
        "total_estimated_hours": sum(item["estimated_hours"] for item in learning_path)
    }
    
    # 6. Confidence Score
    confidence_factors = [
        min(1.0, len(recent_thoughts) / 10),  # More thoughts = higher confidence
        min(1.0, total_skill_mentions / 20),  # More skill mentions = higher confidence
        min(1.0, len(set(thought.get('category', '') for thought in recent_thoughts)) / 5)  # Category diversity
    ]
    confidence_score = min(100, sum(confidence_factors) * 33.33)
    
    # 7. Next Steps
    next_steps = []
    if current_level == "beginner":
        next_steps = [
            "Start with foundational courses in your top skill areas",
            "Practice regularly and log your learning progress",
            "Join communities related to your learning goals"
        ]
    elif current_level == "intermediate":
        next_steps = [
            "Focus on advanced concepts in your strongest areas",
            "Build projects to apply your knowledge",
            "Mentor others to reinforce your learning"
        ]
    else:  # advanced
        next_steps = [
            "Explore cutting-edge topics in your field",
            "Contribute to open source or create content",
            "Consider teaching or consulting opportunities"
        ]
    
    return LearningPathwayInsight(
        current_level=current_level,
        recommended_skills=recommended_skills,
        learning_path=learning_path,
        time_estimates=time_estimates,
        confidence_score=confidence_score,
        next_steps=next_steps
    )

def get_learning_resources(skill: str) -> List[str]:
    """Get learning resources for a specific skill"""
    resources = {
        "programming": [
            "FreeCodeCamp - Web Development",
            "The Odin Project - Full Stack",
            "LeetCode - Algorithm Practice",
            "GitHub - Open Source Projects"
        ],
        "data_science": [
            "Coursera - Data Science Specialization",
            "Kaggle - Data Science Competitions",
            "Fast.ai - Practical Deep Learning",
            "DataCamp - Interactive Courses"
        ],
        "design": [
            "Figma - UI/UX Design",
            "Behance - Design Inspiration",
            "Dribbble - Design Community",
            "Canva - Graphic Design"
        ],
        "business": [
            "Harvard Business Review",
            "Coursera - Business Courses",
            "LinkedIn Learning - Business Skills",
            "Startup School - Entrepreneurship"
        ],
        "writing": [
            "Grammarly - Writing Assistant",
            "Medium - Writing Platform",
            "ProWritingAid - Writing Analysis",
            "MasterClass - Writing Courses"
        ],
        "productivity": [
            "Notion - All-in-One Workspace",
            "Todoist - Task Management",
            "RescueTime - Time Tracking",
            "Forest - Focus App"
        ],
        "creativity": [
            "Pinterest - Creative Inspiration",
            "Behance - Creative Portfolio",
            "Skillshare - Creative Courses",
            "CreativeLive - Live Classes"
        ],
        "communication": [
            "Toastmasters - Public Speaking",
            "Coursera - Communication Courses",
            "LinkedIn Learning - Communication Skills",
            "TED Talks - Speaking Examples"
        ],
        "problem_solving": [
            "LeetCode - Algorithm Problems",
            "HackerRank - Coding Challenges",
            "Project Euler - Mathematical Problems",
            "Brilliant - Math and Science"
        ],
        "learning": [
            "Coursera - Online Courses",
            "edX - University Courses",
            "Khan Academy - Free Education",
            "YouTube - Educational Content"
        ]
    }
    return resources.get(skill, ["General online courses", "Books and articles", "Practice and projects"])

def get_time_estimate(skill: str, current_level: str, target_level: str) -> int:
    """Get time estimate for skill development"""
    base_hours = {
        "beginner_to_intermediate": 40,
        "intermediate_to_advanced": 80
    }
    
    # Adjust based on skill complexity
    skill_complexity = {
        "programming": 1.2,
        "data_science": 1.5,
        "design": 1.0,
        "business": 1.1,
        "writing": 0.8,
        "productivity": 0.7,
        "creativity": 1.0,
        "communication": 0.9,
        "problem_solving": 1.3,
        "learning": 0.8
    }
    
    level_key = f"{current_level}_to_{target_level}"
    base_time = base_hours.get(level_key, 40)
    complexity = skill_complexity.get(skill, 1.0)
    
    return int(base_time * complexity)

def analyze_future_self(thoughts: List[dict]) -> FutureSelfPrediction:
    if not thoughts:
        return FutureSelfPrediction(
            summary="Not enough data to predict your future self. Start logging thoughts to see predictions!",
            predicted_trends=[],
            risk_factors=[],
            positive_outcomes=[],
            advice=["Log thoughts regularly for more accurate predictions."],
            time_horizon="1-3 months"
        )
    
    # Analyze last 30 thoughts or all if less
    recent_thoughts = thoughts[-30:] if len(thoughts) > 30 else thoughts
    
    # Priority trends
    priorities = [t.get('priority', '').lower() for t in recent_thoughts if t.get('priority')]
    urgent_ratio = priorities.count('urgent') / len(priorities) if priorities else 0
    normal_ratio = priorities.count('normal') / len(priorities) if priorities else 0
    low_ratio = priorities.count('low') / len(priorities) if priorities else 0
    
    # Category trends
    categories = [t.get('category', '').lower() for t in recent_thoughts if t.get('category')]
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    
    # Sentiment trends (simple heuristic)
    positive_words = ['happy', 'excited', 'progress', 'success', 'achieved', 'improve', 'enjoy', 'love', 'great', 'wonderful']
    negative_words = ['stress', 'tired', 'worried', 'anxious', 'overwhelmed', 'burnout', 'frustrated', 'exhausted', 'fail', 'problem']
    pos_count = 0
    neg_count = 0
    for t in recent_thoughts:
        content = t.get('content', '').lower()
        pos_count += sum(1 for w in positive_words if w in content)
        neg_count += sum(1 for w in negative_words if w in content)
    total_sentiment = pos_count - neg_count
    
    # Consistency (logging frequency)
    timestamps = [t.get('timestamp') for t in recent_thoughts if t.get('timestamp')]
    days = set()
    for ts in timestamps:
        if not ts:
            continue
        try:
            day = ts.split(',')[0]
            days.add(day)
        except Exception:
            continue
    consistency = len(days) / max(1, (len(recent_thoughts) // 3))  # crude: 1 per 3 thoughts
    
    # Predicted trends
    predicted_trends = []
    if urgent_ratio > 0.4:
        predicted_trends.append("High urgency and fast-paced lifestyle likely to continue")
    if normal_ratio > 0.5:
        predicted_trends.append("Balanced approach to tasks expected")
    if low_ratio > 0.4:
        predicted_trends.append("More relaxed and low-pressure periods ahead")
    if top_categories:
        for cat, count in top_categories:
            if count > 2:
                predicted_trends.append(f"Focus on '{cat.title()}' will shape your near future")
    if total_sentiment > 3:
        predicted_trends.append("Positive mindset and growth likely to continue")
    elif total_sentiment < -3:
        predicted_trends.append("Negative emotions may persist if not addressed")
    if consistency > 0.7:
        predicted_trends.append("Consistent self-reflection will accelerate progress")
    elif consistency < 0.3:
        predicted_trends.append("Irregular logging may slow personal growth")
    
    # Risk factors
    risk_factors = []
    if urgent_ratio > 0.5:
        risk_factors.append("Risk of burnout due to frequent urgent tasks")
    if neg_count > pos_count:
        risk_factors.append("Negative emotions outweigh positives; consider stress management")
    if consistency < 0.3:
        risk_factors.append("Inconsistent self-tracking may hinder improvement")
    if not risk_factors:
        risk_factors.append("No major risks detected")
    
    # Positive outcomes
    positive_outcomes = []
    if pos_count > neg_count:
        positive_outcomes.append("Optimism and progress are likely to grow")
    if top_categories:
        for cat, count in top_categories:
            if cat in ['learning', 'productivity', 'creativity']:
                positive_outcomes.append(f"Skill in '{cat.title()}' will likely improve")
    if consistency > 0.7:
        positive_outcomes.append("Strong self-awareness and habit formation expected")
    if not positive_outcomes:
        positive_outcomes.append("Keep logging thoughts to unlock more positive trends")
    
    # Advice
    advice = []
    if urgent_ratio > 0.5:
        advice.append("Prioritize and delegate urgent tasks to avoid burnout")
    if neg_count > pos_count:
        advice.append("Practice gratitude and positive reframing in your thoughts")
    if consistency < 0.5:
        advice.append("Try to log thoughts more regularly for better predictions")
    if not advice:
        advice.append("Maintain your current habits for continued growth")
    
    # Summary
    if total_sentiment > 3:
        summary = "You are on a positive trajectory. Keep up the good work!"
    elif total_sentiment < -3:
        summary = "You may be experiencing stress or negativity. Consider self-care and support."
    elif urgent_ratio > 0.5:
        summary = "Your future self may face burnout if urgency continues."
    elif consistency > 0.7:
        summary = "Your consistent self-reflection is setting you up for success."
    else:
        summary = "Your future self will reflect your current habits. Stay mindful and intentional."
    
    # Time horizon
    time_horizon = "1-3 months" if len(recent_thoughts) < 15 else "6-12 months"
    
    return FutureSelfPrediction(
        summary=summary,
        predicted_trends=predicted_trends,
        risk_factors=risk_factors,
        positive_outcomes=positive_outcomes,
        advice=advice,
        time_horizon=time_horizon
    )

@app.get("/future-self/", response_model=FutureSelfPrediction)
async def get_future_self():
    """Predict likely future self based on recent thoughts and trends"""
    thoughts = []
    async for thought in thoughts_collection.find():
        thoughts.append(thought_helper(thought))
    return analyze_future_self(thoughts)