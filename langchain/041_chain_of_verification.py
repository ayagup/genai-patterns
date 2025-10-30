"""
Pattern 041: Chain-of-Verification (CoVe)

Description:
    Chain-of-Verification enables agents to reduce hallucinations and improve
    factual accuracy by generating verification questions to check their own
    answers. The agent generates an initial response, creates verification
    questions, answers them, and then revises the original response based on
    the verification results.

Components:
    - Initial Generator: Creates initial response
    - Verification Question Generator: Creates questions to verify response
    - Verification Answerer: Answers verification questions
    - Consistency Checker: Checks for contradictions
    - Response Reviser: Updates response based on verification

Use Cases:
    - Fact-checking automation
    - Reducing hallucinations
    - Improving factual accuracy
    - Knowledge-intensive tasks
    - Research and analysis
    - Educational content

LangChain Implementation:
    Uses multi-step verification chains to check responses, identify
    inconsistencies, and revise outputs for improved accuracy.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class VerificationQuestion:
    """A question to verify part of the response."""
    question: str
    expected_answer: Optional[str] = None
    actual_answer: Optional[str] = None
    is_consistent: Optional[bool] = None
    relevance: float = 1.0


@dataclass
class VerificationResult:
    """Result of verification process."""
    original_response: str
    verification_questions: List[VerificationQuestion]
    inconsistencies: List[str]
    verified_facts: List[str]
    revised_response: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class ChainOfVerificationAgent:
    """
    Agent that verifies its own responses through questioning.
    
    Features:
    - Generates verification questions
    - Checks factual consistency
    - Identifies contradictions
    - Revises responses
    - Improves accuracy
    """
    
    def __init__(
        self,
        response_temperature: float = 0.7,
        verification_temperature: float = 0.3,
        num_verification_questions: int = 5
    ):
        self.response_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=response_temperature)
        self.verification_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=verification_temperature)
        self.num_verification_questions = num_verification_questions
        
        # Initial response prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable assistant. Provide detailed, accurate answers."),
            ("user", "{query}")
        ])
        
        # Verification question generation prompt
        self.verification_questions_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Generate verification questions to check the accuracy of this response.

Create {num_questions} specific, factual questions that verify key claims in the response.
Each question should check a different aspect or claim.

Format:
1. [Question 1]
2. [Question 2]
..."""),
            ("user", "Query: {query}\n\nResponse to verify:\n{response}\n\nGenerate verification questions:")
        ])
        
        # Answer verification questions prompt
        self.answer_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert answering verification questions. Provide concise, factual answers."),
            ("user", "Question: {question}\n\nProvide a factual answer:")
        ])
        
        # Consistency check prompt
        self.consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", """Compare the original response with the verification answers.

Identify:
1. Inconsistencies or contradictions
2. Verified facts that are consistent
3. Overall confidence in the original response (0.0-1.0)

Format:
INCONSISTENCIES:
- [List any inconsistencies]

VERIFIED_FACTS:
- [List verified facts]

CONFIDENCE: [0.0-1.0]"""),
            ("user", """Original Response:
{original_response}

Verification Q&A:
{verification_qa}

Analyze consistency:""")
        ])
        
        # Revision prompt
        self.revision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an editor improving response accuracy.

Revise the original response to:
1. Fix any inconsistencies identified
2. Maintain verified facts
3. Remove or correct uncertain information
4. Keep the response helpful and complete"""),
            ("user", """Query: {query}

Original Response:
{original_response}

Issues Found:
{inconsistencies}

Provide revised response:""")
        ])
        
        # Verification history
        self.verifications: List[VerificationResult] = []
    
    def generate_response(self, query: str) -> str:
        """Generate initial response to query."""
        chain = self.response_prompt | self.response_llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def generate_verification_questions(
        self,
        query: str,
        response: str
    ) -> List[VerificationQuestion]:
        """
        Generate questions to verify the response.
        
        Returns:
            List of VerificationQuestion objects
        """
        chain = self.verification_questions_prompt | self.verification_llm | StrOutputParser()
        questions_text = chain.invoke({
            "query": query,
            "response": response,
            "num_questions": self.num_verification_questions
        })
        
        # Parse questions
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                question = line.split('.', 1)[-1].strip()
                if question.startswith('-'):
                    question = question[1:].strip()
                
                if question:
                    questions.append(VerificationQuestion(question=question))
        
        return questions[:self.num_verification_questions]
    
    def answer_verification_questions(
        self,
        questions: List[VerificationQuestion]
    ) -> List[VerificationQuestion]:
        """
        Answer verification questions independently.
        
        Returns:
            List of VerificationQuestion objects with answers
        """
        chain = self.answer_verification_prompt | self.verification_llm | StrOutputParser()
        
        for question in questions:
            answer = chain.invoke({"question": question.question})
            question.actual_answer = answer
        
        return questions
    
    def check_consistency(
        self,
        original_response: str,
        questions: List[VerificationQuestion]
    ) -> Tuple[List[str], List[str], float]:
        """
        Check consistency between original response and verification answers.
        
        Returns:
            Tuple of (inconsistencies, verified_facts, confidence)
        """
        # Format verification Q&A
        verification_qa = "\n\n".join([
            f"Q: {q.question}\nA: {q.actual_answer}"
            for q in questions if q.actual_answer
        ])
        
        chain = self.consistency_prompt | self.verification_llm | StrOutputParser()
        result = chain.invoke({
            "original_response": original_response,
            "verification_qa": verification_qa
        })
        
        # Parse result
        inconsistencies = []
        verified_facts = []
        confidence = 0.5
        
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("INCONSISTENCIES:"):
                current_section = "inconsistencies"
            elif line.startswith("VERIFIED_FACTS:"):
                current_section = "verified_facts"
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
                current_section = None
            elif line and line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section == "inconsistencies":
                    inconsistencies.append(item)
                elif current_section == "verified_facts":
                    verified_facts.append(item)
        
        return inconsistencies, verified_facts, confidence
    
    def revise_response(
        self,
        query: str,
        original_response: str,
        inconsistencies: List[str]
    ) -> str:
        """
        Revise response to fix inconsistencies.
        
        Returns:
            Revised response
        """
        if not inconsistencies:
            return original_response
        
        inconsistencies_text = "\n".join([f"- {inc}" for inc in inconsistencies])
        
        chain = self.revision_prompt | self.response_llm | StrOutputParser()
        revised = chain.invoke({
            "query": query,
            "original_response": original_response,
            "inconsistencies": inconsistencies_text
        })
        
        return revised
    
    def verify_response(
        self,
        query: str,
        response: Optional[str] = None
    ) -> VerificationResult:
        """
        Complete Chain-of-Verification process.
        
        Steps:
        1. Generate initial response (or use provided)
        2. Generate verification questions
        3. Answer verification questions
        4. Check consistency
        5. Revise if needed
        
        Args:
            query: The query to answer
            response: Optional pre-generated response
            
        Returns:
            VerificationResult object
        """
        # Step 1: Generate or use provided response
        if response is None:
            response = self.generate_response(query)
        
        original_response = response
        
        # Step 2: Generate verification questions
        questions = self.generate_verification_questions(query, response)
        
        # Step 3: Answer verification questions
        questions = self.answer_verification_questions(questions)
        
        # Step 4: Check consistency
        inconsistencies, verified_facts, confidence = self.check_consistency(
            original_response,
            questions
        )
        
        # Step 5: Revise if inconsistencies found
        if inconsistencies:
            revised_response = self.revise_response(query, original_response, inconsistencies)
        else:
            revised_response = original_response
        
        result = VerificationResult(
            original_response=original_response,
            verification_questions=questions,
            inconsistencies=inconsistencies,
            verified_facts=verified_facts,
            revised_response=revised_response,
            confidence_score=confidence
        )
        
        self.verifications.append(result)
        
        return result
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get statistics about verifications."""
        if not self.verifications:
            return {"total_verifications": 0}
        
        total_questions = sum(len(v.verification_questions) for v in self.verifications)
        total_inconsistencies = sum(len(v.inconsistencies) for v in self.verifications)
        avg_confidence = sum(v.confidence_score for v in self.verifications) / len(self.verifications)
        revision_rate = sum(1 for v in self.verifications if v.revised_response != v.original_response) / len(self.verifications)
        
        return {
            "total_verifications": len(self.verifications),
            "total_questions_generated": total_questions,
            "total_inconsistencies_found": total_inconsistencies,
            "average_confidence": avg_confidence,
            "revision_rate": revision_rate
        }


def demonstrate_chain_of_verification():
    """
    Demonstrates Chain-of-Verification for reducing hallucinations.
    """
    print("=" * 80)
    print("CHAIN-OF-VERIFICATION DEMONSTRATION")
    print("=" * 80)
    
    # Create CoVe agent
    agent = ChainOfVerificationAgent(
        response_temperature=0.7,
        verification_temperature=0.3,
        num_verification_questions=4
    )
    
    # Test 1: Factual query with potential for errors
    print("\n" + "=" * 80)
    print("Test 1: Historical Facts Verification")
    print("=" * 80)
    
    query1 = "When did World War II end and what were the key events?"
    print(f"\nQuery: {query1}")
    
    result1 = agent.verify_response(query1)
    
    print("\n[Original Response]")
    print(result1.original_response[:300] + "..." if len(result1.original_response) > 300 else result1.original_response)
    
    print("\n[Verification Questions]")
    for i, q in enumerate(result1.verification_questions, 1):
        print(f"{i}. {q.question}")
        if q.actual_answer:
            print(f"   Answer: {q.actual_answer[:100]}...")
    
    if result1.inconsistencies:
        print("\n[Inconsistencies Found]")
        for inc in result1.inconsistencies:
            print(f"  - {inc}")
    
    if result1.verified_facts:
        print("\n[Verified Facts]")
        for fact in result1.verified_facts[:3]:
            print(f"  - {fact}")
    
    print(f"\n[Confidence Score]: {result1.confidence_score:.2f}")
    
    if result1.revised_response != result1.original_response:
        print("\n[Revised Response]")
        print(result1.revised_response[:300] + "..." if len(result1.revised_response) > 300 else result1.revised_response)
    else:
        print("\n[No Revision Needed]")
    
    # Test 2: Scientific query
    print("\n" + "=" * 80)
    print("Test 2: Scientific Facts Verification")
    print("=" * 80)
    
    query2 = "How does photosynthesis work in plants?"
    print(f"\nQuery: {query2}")
    
    result2 = agent.verify_response(query2)
    
    print("\n[Original Response]")
    print(result2.original_response[:250] + "..." if len(result2.original_response) > 250 else result2.original_response)
    
    print("\n[Verification Questions Generated]")
    for i, q in enumerate(result2.verification_questions, 1):
        print(f"{i}. {q.question}")
    
    print(f"\n[Verification Results]")
    print(f"Inconsistencies Found: {len(result2.inconsistencies)}")
    print(f"Verified Facts: {len(result2.verified_facts)}")
    print(f"Confidence Score: {result2.confidence_score:.2f}")
    
    # Test 3: Potentially tricky query
    print("\n" + "=" * 80)
    print("Test 3: Complex Query with Verification")
    print("=" * 80)
    
    query3 = "What are the main differences between machine learning and deep learning?"
    print(f"\nQuery: {query3}")
    
    result3 = agent.verify_response(query3)
    
    print("\n[Verification Process]")
    print(f"Questions Generated: {len(result3.verification_questions)}")
    print(f"Inconsistencies: {len(result3.inconsistencies)}")
    print(f"Verified Facts: {len(result3.verified_facts)}")
    print(f"Confidence: {result3.confidence_score:.2f}")
    print(f"Response Revised: {'Yes' if result3.revised_response != result3.original_response else 'No'}")
    
    print("\n[Sample Verification Q&A]")
    for q in result3.verification_questions[:2]:
        print(f"\nQ: {q.question}")
        print(f"A: {q.actual_answer[:150]}..." if q.actual_answer and len(q.actual_answer) > 150 else f"A: {q.actual_answer}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("Verification Statistics")
    print("=" * 80)
    
    stats = agent.get_verification_statistics()
    print(f"\nTotal Verifications: {stats['total_verifications']}")
    print(f"Total Questions Generated: {stats['total_questions_generated']}")
    print(f"Total Inconsistencies Found: {stats['total_inconsistencies_found']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    print(f"Revision Rate: {stats['revision_rate']:.1%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Chain-of-Verification provides:
✓ Self-verification through questioning
✓ Inconsistency detection
✓ Fact checking automation
✓ Hallucination reduction
✓ Improved factual accuracy
✓ Transparent verification process

This pattern excels at:
- Fact-checking responses
- Reducing hallucinations
- Knowledge-intensive tasks
- Research and analysis
- Educational content
- High-stakes information

Verification process:
1. Generate initial response
2. Create verification questions
3. Answer questions independently
4. Check consistency
5. Identify contradictions
6. Revise response if needed

Verification components:
- Initial Generator: Creates response
- Question Generator: Creates verification questions
- Answer Generator: Answers questions independently
- Consistency Checker: Finds contradictions
- Response Reviser: Fixes inconsistencies

Question types:
- Factual verification: Check specific facts
- Logical consistency: Check reasoning
- Completeness: Check coverage
- Accuracy: Check precision

Benefits:
- Accuracy: Catch factual errors
- Reliability: Reduce hallucinations
- Transparency: Show verification process
- Self-correction: Auto-fix errors
- Confidence: Measure certainty
- Scalable: Works for any domain

Verification strategies:
- Independent Q&A: Answer questions separately
- Consistency checking: Compare answers
- Contradiction detection: Find conflicts
- Confidence scoring: Quantify certainty
- Iterative revision: Fix issues

Use Chain-of-Verification when:
- Factual accuracy is critical
- Hallucinations are a concern
- Need transparent fact-checking
- Working with knowledge-intensive tasks
- Building educational systems
- High-stakes decision making

Comparison with other patterns:
- vs Self-Evaluation: CoVe focuses on factual verification
- vs Constitutional AI: CoVe checks facts, not values
- vs Feedback Loops: CoVe is immediate, not learning-based
""")


if __name__ == "__main__":
    demonstrate_chain_of_verification()
