#!/usr/bin/env python3
"""
ModelMuxer Budget Dashboard

Interactive dashboard for monitoring and managing budgets in real-time.
Provides cost analytics, budget status, and budget management capabilities.

Usage:
    python scripts/budget_dashboard.py                    # Show dashboard
    python scripts/budget_dashboard.py set daily 5.0      # Set daily budget
    python scripts/budget_dashboard.py set weekly 25.0    # Set weekly budget
    python scripts/budget_dashboard.py help               # Show help

Requires ModelMuxer to be running on localhost:8000
"""
import requests  # type: ignore
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_KEY = "sk-test-claude-dev"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def get_budget_status():
    """Get current budget status"""
    try:
        response = requests.get(f"{BASE_URL}/v1/analytics/budgets", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting budget status: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_cost_analytics(days=7):
    """Get cost analytics"""
    try:
        response = requests.get(f"{BASE_URL}/v1/analytics/costs?days={days}", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting cost analytics: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def set_budget(budget_type, limit, provider=None, model=None, thresholds=None):
    """Set a new budget"""
    data = {"budget_type": budget_type, "budget_limit": float(limit)}

    if provider:
        data["provider"] = provider
    if model:
        data["model"] = model
    if thresholds:
        data["alert_thresholds"] = thresholds

    try:
        response = requests.post(
            f"{BASE_URL}/v1/analytics/budgets", headers=get_headers(), data=json.dumps(data)
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error setting budget: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def display_dashboard():
    """Display the budget dashboard"""
    print("=" * 60)
    print("🏦 MODELMUXER BUDGET DASHBOARD")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get budget status
    budget_status = get_budget_status()
    if not budget_status:
        print("❌ Unable to retrieve budget status")
        return

    budgets = budget_status.get("budgets", [])

    if not budgets:
        print("📊 No budgets configured")
        print()
        print("💡 To set up a budget, use:")
        print("   python budget_dashboard.py set daily 1.0")
        return

    print(f"📊 ACTIVE BUDGETS ({len(budgets)} configured)")
    print("-" * 60)

    for budget in budgets:
        budget_type = budget["budget_type"].upper()
        limit = budget["budget_limit"]
        usage = budget["current_usage"]
        percentage = budget["usage_percentage"]
        remaining = budget["remaining_budget"]

        # Status icon
        if percentage > 95:
            status = "🔴 CRITICAL"
        elif percentage > 80:
            status = "🟠 WARNING"
        elif percentage > 50:
            status = "🟡 CAUTION"
        else:
            status = "🟢 HEALTHY"

        print(f"📈 {budget_type} BUDGET {status}")
        print(f"   💰 Limit: ${limit:,.6f}")
        print(f"   💸 Used: ${usage:,.6f} ({percentage:.3f}%)")
        print(f"   💵 Remaining: ${remaining:,.6f}")

        if budget["provider"]:
            print(f"   🏢 Provider: {budget['provider']}")
        if budget["model"]:
            print(f"   🤖 Model: {budget['model']}")

        print(f"   📅 Period: {budget['period_start']} to {budget['period_end']}")

        if budget["alerts"]:
            print(f"   ⚠️  Alerts: {len(budget['alerts'])} active")

        print()

    # Get cost analytics
    print("📈 COST ANALYTICS")
    print("-" * 60)
    analytics = get_cost_analytics()
    if analytics:
        print(f"📊 Total Requests: {analytics.get('total_requests', 0)}")
        print(f"💰 Total Cost: ${analytics.get('total_cost', 0):.6f}")

        cost_by_provider = analytics.get("cost_by_provider", {})
        if cost_by_provider:
            print("\n🏢 Cost by Provider:")
            for provider, cost in cost_by_provider.items():
                print(f"   {provider}: ${cost:.6f}")

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "set" and len(sys.argv) >= 4:
            budget_type = sys.argv[2]
            budget_limit = float(sys.argv[3])

            provider = sys.argv[4] if len(sys.argv) > 4 else None
            model = sys.argv[5] if len(sys.argv) > 5 else None

            result = set_budget(budget_type, budget_limit, provider, model)
            if result:
                print(f"✅ Budget set successfully: {budget_type} ${budget_limit}")
                if provider:
                    print(f"   Provider: {provider}")
                if model:
                    print(f"   Model: {model}")
            else:
                print("❌ Failed to set budget")
            return

        elif command == "help":
            print("ModelMuxer Budget Dashboard Commands:")
            print("  python budget_dashboard.py                    - Show dashboard")
            print("  python budget_dashboard.py set daily 1.0      - Set daily budget")
            print(
                "  python budget_dashboard.py set weekly 5.0 openai - Set weekly budget for openai"
            )
            print("  python budget_dashboard.py help               - Show this help")
            return

    # Default: show dashboard
    display_dashboard()


if __name__ == "__main__":
    main()
