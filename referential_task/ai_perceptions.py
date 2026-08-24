"""AI-vs-AI partner perceptions for the basket referential task."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Player


AI_VS_AI_PERCEPTION_PROMPT = """
You just completed a collaborative task with an AI partner across 5 rounds.
In this task, you and your partner had to work together to correctly order 12 baskets.

Based on the complete conversation history below, please evaluate your AI partner.

For each statement, provide a rating from 1 to 5 where:
1 = strongly disagree
2 = disagree
3 = neutral
4 = agree
5 = strongly agree

Please respond with a JSON object in the following format:
{
    "partner_capable": <1-5>,
    "partner_helpful": <1-5>,
    "partner_understood": <1-5>,
    "partner_adapted": <1-5>,
    "collaboration_improved": <1-5>,
    "partner_comment": "<your comment about how your partner performed>"
}

The questions are:
- partner_capable: "My partner was capable of doing their task"
- partner_helpful: "My partner was helpful to me for completing my task"
- partner_understood: "My partner understood what I was trying to communicate"
- partner_adapted: "My partner adapted to the way I communicated over time"
- collaboration_improved: "Our collaboration improved over time"
- partner_comment: "Please comment about how your partner did the task"

Be honest and thoughtful in your evaluation based on the conversation history.
""".strip()

def generate_ai_vs_ai_perceptions(player: "Player") -> bool:
    """Generate mutual AI perceptions (Director evaluating Matcher, and Matcher evaluating Director).
    
    This is called automatically in AI-vs-AI mode at the end of the final round.
    """
    from .ai_vs_ai import _build_api_call_kwargs, _get_ai_client
    client = _get_ai_client()
    if client is None:
        logging.warning("[AI_VS_AI_PERCEPTIONS] No OpenAI client available")
        return False

    try:
        # Gather complete chat history across all rounds
        all_messages = []
        if hasattr(player, "in_all_rounds"):
            all_players = player.in_all_rounds()
        else:
            all_players = [player]

        for round_player in all_players:
            round_num = round_player.round_number
            try:
                ai_msgs = json.loads(round_player.group.ai_messages or "[]")
                for msg in ai_msgs:
                    if isinstance(msg, dict):
                        msg["round"] = round_num
                        all_messages.append(msg)
            except Exception:
                pass

        if not all_messages:
            logging.warning("[AI_VS_AI_PERCEPTIONS] No messages found in history")
            return False

        # Build conversation text
        conversation_text_base = "=== COMPLETE CONVERSATION HISTORY ===\n\n"
        current_round = 0
        for msg in all_messages:
            msg_round = msg.get("round", 0)
            if msg_round != current_round:
                current_round = msg_round
                conversation_text_base += f"\n--- Round {current_round} ---\n\n"

            sender = msg.get("sender_role", "unknown")
            text = msg.get("text", "")
            if text:
                conversation_text_base += f"{sender.upper()}: {text}\n"

        def evaluate_partner(evaluator_role: str, partner_role: str):
            prefix = f"Your role in the task was: {evaluator_role.upper()}\n"
            prefix += f"Your partner's role was: {partner_role.upper()}\n\n"
            conversation_text = prefix + conversation_text_base
            
            messages = [
                {"role": "system", "content": AI_VS_AI_PERCEPTION_PROMPT},
                {"role": "user", "content": conversation_text},
            ]
            
            model = player.session.config.get(f"ai_{evaluator_role}_model", "gpt-4o-mini")
            
            api_kwargs = _build_api_call_kwargs(
                model=model,
                messages=messages,
                player=player,
                max_tokens=600,
            )
            response = client.chat.completions.create(**api_kwargs)
            
            reply = response.choices[0].message.content
            if not reply: return None
            
            reply = reply.strip()
            if "```json" in reply:
                reply = reply[reply.find("```json")+7:reply.rfind("```")].strip()
            elif "```" in reply:
                reply = reply[reply.find("```")+3:reply.rfind("```")].strip()
                
            perceptions = json.loads(reply)
            result = {}
            for key in ["partner_capable", "partner_helpful", "partner_understood", "partner_adapted", "collaboration_improved"]:
                val = perceptions.get(key)
                if isinstance(val, (int, float)):
                    result[key] = max(1, min(5, int(val)))
                else:
                    result[key] = 3
            result["partner_comment"] = str(perceptions.get("partner_comment", ""))[:2000]
            result["_raw"] = perceptions
            return result

        # 1. Director evaluating Matcher
        logging.info("[AI_VS_AI_PERCEPTIONS] Fetching Director's perceptions...")
        dir_eval = evaluate_partner("director", "matcher")
        if dir_eval:
            group = player.group
            group.ai_director_partner_capable = dir_eval["partner_capable"]
            group.ai_director_partner_helpful = dir_eval["partner_helpful"]
            group.ai_director_partner_understood = dir_eval["partner_understood"]
            group.ai_director_partner_adapted = dir_eval["partner_adapted"]
            group.ai_director_collaboration_improved = dir_eval["collaboration_improved"]
            group.ai_director_partner_comment = dir_eval["partner_comment"]
            group.ai_director_perceptions_raw = json.dumps(dir_eval["_raw"])

        # 2. Matcher evaluating Director
        logging.info("[AI_VS_AI_PERCEPTIONS] Fetching Matcher's perceptions...")
        mat_eval = evaluate_partner("matcher", "director")
        if mat_eval:
            group = player.group
            group.ai_matcher_partner_capable = mat_eval["partner_capable"]
            group.ai_matcher_partner_helpful = mat_eval["partner_helpful"]
            group.ai_matcher_partner_understood = mat_eval["partner_understood"]
            group.ai_matcher_partner_adapted = mat_eval["partner_adapted"]
            group.ai_matcher_collaboration_improved = mat_eval["collaboration_improved"]
            group.ai_matcher_partner_comment = mat_eval["partner_comment"]
            group.ai_matcher_perceptions_raw = json.dumps(mat_eval["_raw"])

        return True
    except Exception as e:
        import traceback
        logging.error("[AI_VS_AI_PERCEPTIONS] Error generating AI vs AI perceptions: %s\n%s", e, traceback.format_exc())
        return False
