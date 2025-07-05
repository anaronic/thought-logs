#!/usr/bin/env python3
"""
Test script for Learning Pathway Optimizer
"""

import asyncio
from datetime import datetime, timedelta
from main import analyze_learning_pathway

# Test data with different skill focuses
test_thoughts_programming = [
    {
        "id": "1",
        "content": "Working on a new Python project. Need to implement the API endpoints and database connections.",
        "category": "Programming",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "2", 
        "content": "Debugging the React frontend. The state management is getting complex with Redux.",
        "category": "Programming",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "3",
        "content": "Learning about machine learning algorithms. The data preprocessing is crucial for model performance.",
        "category": "Data Science",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "4",
        "content": "Need to optimize the database queries. The performance is slow with large datasets.",
        "category": "Programming",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=4)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "5",
        "content": "Working on UI/UX design for the mobile app. The user experience needs improvement.",
        "category": "Design",
        "priority": "low",
        "timestamp": (datetime.now() - timedelta(hours=5)).strftime("%B %d, %Y, %I:%M %p")
    }
]

test_thoughts_business = [
    {
        "id": "1",
        "content": "Analyzing market trends for our startup. Need to understand customer behavior patterns.",
        "category": "Business",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "2", 
        "content": "Preparing for investor pitch. The financial projections need to be more detailed.",
        "category": "Business",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "3",
        "content": "Working on marketing strategy. Social media campaigns are showing good engagement.",
        "category": "Business",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "4",
        "content": "Team leadership meeting. Need to improve communication between departments.",
        "category": "Communication",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=4)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "5",
        "content": "Product development planning. Customer feedback is very positive.",
        "category": "Business",
        "priority": "low",
        "timestamp": (datetime.now() - timedelta(hours=5)).strftime("%B %d, %Y, %I:%M %p")
    }
]

test_thoughts_creative = [
    {
        "id": "1",
        "content": "Working on a new creative project. The design inspiration is coming from nature.",
        "category": "Creativity",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "2", 
        "content": "Writing a blog post about creative process. Need to improve my writing skills.",
        "category": "Writing",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "3",
        "content": "Learning new art techniques. The digital painting tools are amazing.",
        "category": "Creativity",
        "priority": "low",
        "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "4",
        "content": "Brainstorming ideas for the next project. Innovation is key to success.",
        "category": "Creativity",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=4)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "5",
        "content": "Working on music composition. The melody is starting to take shape.",
        "category": "Creativity",
        "priority": "low",
        "timestamp": (datetime.now() - timedelta(hours=5)).strftime("%B %d, %Y, %I:%M %p")
    }
]

async def test_learning_pathway():
    print("🎯 Testing Learning Pathway Optimizer")
    print("=" * 60)
    
    # Test with no thoughts
    print("\n1. Testing with no thoughts:")
    result = analyze_learning_pathway([])
    print(f"Current Level: {result.current_level}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Recommended Skills: {result.recommended_skills}")
    print(f"Next Steps: {result.next_steps}")
    
    # Test with programming-focused thoughts
    print("\n2. Testing with programming-focused thoughts:")
    result = analyze_learning_pathway(test_thoughts_programming)
    print(f"Current Level: {result.current_level}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Recommended Skills: {result.recommended_skills}")
    print(f"Learning Path:")
    for path in result.learning_path:
        print(f"  - {path['skill']}: {path['estimated_hours']}h ({path['priority']} priority)")
    print(f"Time Estimates: {result.time_estimates}")
    print(f"Next Steps: {result.next_steps}")
    
    # Test with business-focused thoughts
    print("\n3. Testing with business-focused thoughts:")
    result = analyze_learning_pathway(test_thoughts_business)
    print(f"Current Level: {result.current_level}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Recommended Skills: {result.recommended_skills}")
    print(f"Learning Path:")
    for path in result.learning_path:
        print(f"  - {path['skill']}: {path['estimated_hours']}h ({path['priority']} priority)")
    print(f"Time Estimates: {result.time_estimates}")
    print(f"Next Steps: {result.next_steps}")
    
    # Test with creative-focused thoughts
    print("\n4. Testing with creative-focused thoughts:")
    result = analyze_learning_pathway(test_thoughts_creative)
    print(f"Current Level: {result.current_level}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Recommended Skills: {result.recommended_skills}")
    print(f"Learning Path:")
    for path in result.learning_path:
        print(f"  - {path['skill']}: {path['estimated_hours']}h ({path['priority']} priority)")
    print(f"Time Estimates: {result.time_estimates}")
    print(f"Next Steps: {result.next_steps}")
    
    # Test with mixed thoughts (advanced level)
    print("\n5. Testing with mixed thoughts (advanced level):")
    mixed_thoughts = test_thoughts_programming + test_thoughts_business + test_thoughts_creative
    result = analyze_learning_pathway(mixed_thoughts)
    print(f"Current Level: {result.current_level}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Recommended Skills: {result.recommended_skills}")
    print(f"Learning Path:")
    for path in result.learning_path:
        print(f"  - {path['skill']}: {path['estimated_hours']}h ({path['priority']} priority)")
    print(f"Time Estimates: {result.time_estimates}")
    print(f"Next Steps: {result.next_steps}")

if __name__ == "__main__":
    asyncio.run(test_learning_pathway()) 