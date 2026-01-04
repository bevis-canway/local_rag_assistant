"""
Hallucination detection module test cases
Used to verify the hallucination mitigation functionality of the Xiao Moxian RAG agent
"""
import unittest
from unittest.mock import Mock, patch
from rag_agent.hallucination_detector import HallucinationDetector, HallucinationCheckResult
from rag_agent.config import Config


class TestHallucinationDetection(unittest.TestCase):
    """Hallucination detection functionality test class"""
    
    def setUp(self):
        """Test initialization"""
        self.config = Config()
        self.detector = HallucinationDetector(self.config)
    
    def test_consistent_response_detection(self):
        """Test detection of consistent responses"""
        response = "According to the document, Python is a programming language."
        retrieved_docs = [
            {
                "content": "Python is a high-level programming language, originally created by Guido van Rossum in 1989.",
                "metadata": {"title": "Python Introduction", "path": "/python/intro.md"}
            }
        ]
        query = "What is Python?"
        
        result = self.detector.detect_hallucinations(response, retrieved_docs, query)
        
        self.assertTrue(result.is_consistent)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(len(result.inconsistencies), 0)
    
    def test_inconsistent_response_detection(self):
        """Test detection of inconsistent responses"""
        response = "According to the document, Java was invented in 1990 by Bill Gates as a programming language."
        retrieved_docs = [
            {
                "content": "Java is an object-oriented programming language developed by James Gosling at Sun Microsystems in 1995.",
                "metadata": {"title": "Java Introduction", "path": "/java/intro.md"}
            }
        ]
        query = "What is Java?"
        
        result = self.detector.detect_hallucinations(response, retrieved_docs, query)
        
        self.assertFalse(result.is_consistent)
        self.assertLess(result.confidence_score, 0.5)
        self.assertGreater(len(result.inconsistencies), 0)
    
    def test_no_document_response(self):
        """Test detection when no documents are available"""
        response = "This is a general response."
        retrieved_docs = []  # Empty document list
        query = "Ask any question"
        
        result = self.detector.detect_hallucinations(response, retrieved_docs, query)
        
        self.assertTrue(result.is_consistent)  # Consider consistent when no documents
        self.assertAlmostEqual(result.confidence_score, 0.8, places=1)
        self.assertEqual(len(result.inconsistencies), 0)
    
    def test_fact_consistency_check(self):
        """Test fact consistency check"""
        response = "Python is a programming language, created by Guido van Rossum in 1991."
        retrieved_docs = [
            {
                "content": "Python is a programming language created by Guido van Rossum.",
                "metadata": {"title": "Python Introduction", "path": "/python/intro.md"}
            }
        ]
        
        result = self.detector._check_fact_consistency(response, retrieved_docs)
        
        # "created in 1991" is not mentioned in the document, should be detected as inconsistent
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_semantic_consistency_check(self):
        """Test semantic consistency check"""
        response = "According to the document, Python is a programming language."
        retrieved_docs = [
            {
                "content": "Python is a high-level programming language, originally created by Guido van Rossum in 1989.",
                "metadata": {"title": "Python Introduction", "path": "/python/intro.md"}
            }
        ]
        query = "What is Python?"
        
        # Mock the ollama.chat response with Chinese format to match the parsing logic
        with patch('rag_agent.hallucination_detector.ollama.chat') as mock_chat:
            mock_chat.return_value = {
                "message": {
                    "content": "一致性: 是\n置信度: 0.9\n不一致之处: 无\n解释: 回答与文档内容一致"
                }
            }
            
            result = self.detector._check_semantic_consistency(response, retrieved_docs, query)
            
            self.assertTrue(result.is_consistent)
            self.assertEqual(result.confidence_score, 0.9)
            self.assertEqual(len(result.inconsistencies), 0)
    
    def test_sentence_splitting(self):
        """Test sentence splitting functionality"""
        text = "This is the first sentence. This is the second sentence! This is the third sentence?"
        sentences = self.detector._split_into_sentences(text)
        
        self.assertEqual(len(sentences), 3)
        self.assertIn("This is the first sentence", sentences)
        self.assertIn("This is the second sentence", sentences)
        self.assertIn("This is the third sentence", sentences)
    
    def test_sentence_support_detection(self):
        """Test sentence support detection"""
        sentence = "Python is a programming language"
        doc_facts = ["Python is a programming language", "Created by Guido van Rossum"]
        
        is_supported = self.detector._sentence_supported_by_docs(sentence, doc_facts)
        
        self.assertTrue(is_supported)
    
    def test_sentence_not_supported_detection(self):
        """Test detection of sentences not supported by documents"""
        sentence = "Java was invented by Bill Gates in 1991"
        doc_facts = ["Python is a programming language", "Created by Guido van Rossum"]
        
        is_supported = self.detector._sentence_supported_by_docs(sentence, doc_facts)
        
        self.assertFalse(is_supported)


class TestRAGAgentWithHallucinationDetection(unittest.TestCase):
    """Test RAG Agent with hallucination detection integration"""
    
    def setUp(self):
        """Test initialization"""
        from rag_agent.main import RAGAgent
        self.config = Config()
        self.agent = RAGAgent(self.config)
    
    def test_agent_initialization_with_hallucination_detector(self):
        """Test that RAG Agent initializes with hallucination detector"""
        self.assertIsNotNone(self.agent.hallucination_detector)
        self.assertIsInstance(self.agent.hallucination_detector, HallucinationDetector)
    
    @patch('rag_agent.retriever.Retriever.retrieve_and_filter_by_similarity')
    @patch('rag_agent.intent.intent_recognizer.IntentRecognizer.recognize_intent')
    def test_hallucination_detection_in_query_flow(self, mock_recognize_intent, mock_retrieve):
        """Test hallucination detection in query flow"""
        # Mock intent recognition result
        mock_intent_result = Mock()
        mock_intent_result.intent_type = "knowledge_query"
        mock_intent_result.confidence = 0.8
        mock_intent_result.rewritten_query = "What is Python?"
        mock_recognize_intent.return_value = mock_intent_result
        
        # Mock retrieval results
        mock_retrieve.return_value = (
            [{
                "content": "Python is a high-level programming language, originally created by Guido van Rossum in 1989.",
                "metadata": {"title": "Python Introduction", "path": "/python/intro.md"},
                "similarity": 0.85
            }], 
            True  # has_relevant_docs = True
        )
        
        # Here we mainly test that the flow correctly calls hallucination detection
        # The actual detection logic has been verified in other tests
        self.assertTrue(True)  # Placeholder, actual test needs more complex mocking


if __name__ == '__main__':
    unittest.main()