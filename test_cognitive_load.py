#!/usr/bin/env python3
"""
Test script for Cognitive Load Analysis
"""

import asyncio
import motor.motor_asyncio
from datetime import datetime, timedelta
from main import analyze_cognitive_load

# Test data
test_thoughts = [
    {
        "id": "1",
        "content": "I'm feeling really stressed about the upcoming deadline. There's so much pressure to deliver this project on time and I'm worried I won't be able to meet the expectations.",
        "category": "Work",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "2", 
        "content": "Need to remember to call the client about the requirements. This is very important and I can't forget about it.",
        "category": "Work",
        "priority": "urgent",
        "timestamp": (datetime.now() - timedelta(hours=0.5)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "3",
        "content": "Thinking about dinner plans for tonight. Should I cook or order takeout?",
        "category": "Personal",
        "priority": "low",
        "timestamp": (datetime.now() - timedelta(hours=0.25)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "4",
        "content": "The meeting went really well today! I'm excited about the progress we made on the project.",
        "category": "Work",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%B %d, %Y, %I:%M %p")
    },
    {
        "id": "5",
        "content": "I'm exhausted from all the work today. Need to take a break and relax.",
        "category": "Personal",
        "priority": "normal",
        "timestamp": (datetime.now() - timedelta(hours=3)).strftime("%B %d, %Y, %I:%M %p")
    }
]

async def test_cognitive_load():
    print("🧠 Testing Cognitive Load Analysis")
    print("=" * 50)
    
    # Test with no thoughts
    print("\n1. Testing with no thoughts:")
    result = analyze_cognitive_load([])
    print(f"Score: {result.score}")
    print(f"Level: {result.level}")
    print(f"Indicators: {result.indicators}")
    print(f"Suggestions: {result.suggestions}")
    
    # Test with few thoughts
    print("\n2. Testing with few thoughts (2):")
    result = analyze_cognitive_load(test_thoughts[:2])
    print(f"Score: {result.score}")
    print(f"Level: {result.level}")
    print(f"Indicators: {result.indicators}")
    print(f"Suggestions: {result.suggestions}")
    print(f"Patterns: {result.patterns}")
    
    # Test with more thoughts
    print("\n3. Testing with more thoughts (5):")
    result = analyze_cognitive_load(test_thoughts)
    print(f"Score: {result.score}")
    print(f"Level: {result.level}")
    print(f"Indicators: {result.indicators}")
    print(f"Suggestions: {result.suggestions}")
    print(f"Patterns: {result.patterns}")
    
    # Test with low-stress thoughts
    print("\n4. Testing with low-stress thoughts:")
    low_stress_thoughts = [
        {
            "id": "1",
            "content": "Had a great day today!",
            "category": "Personal",
            "priority": "low",
            "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%B %d, %Y, %I:%M %p")
        },
        {
            "id": "2",
            "content": "Looking forward to the weekend.",
            "category": "Personal", 
            "priority": "low",
            "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%B %d, %Y, %I:%M %p")
        }
    ]
    result = analyze_cognitive_load(low_stress_thoughts)
    print(f"Score: {result.score}")
    print(f"Level: {result.level}")
    print(f"Indicators: {result.indicators}")
    print(f"Suggestions: {result.suggestions}")

if __name__ == "__main__":
    asyncio.run(test_cognitive_load()) 