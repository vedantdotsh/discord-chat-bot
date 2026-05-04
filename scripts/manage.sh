#!/bin/bash

SERVICE="${SERVICE:-discord-bot}"
BOT_DIR="${BOT_DIR:-/home/discordbot/discord-chat-bot}"

case "$1" in
    start)
        echo "Starting bot..."
        systemctl start "$SERVICE"
        systemctl status "$SERVICE" --no-pager ;;

    stop)
        echo "Stopping bot..."
        systemctl stop "$SERVICE"
        echo "Bot stopped." ;;

    restart)
        echo "Restarting bot..."
        systemctl restart "$SERVICE"
        systemctl status "$SERVICE" --no-pager ;;

    status)
        systemctl status "$SERVICE" --no-pager ;;

    logs)
        echo "Streaming live logs. Press Ctrl+C to stop."
        journalctl -u "$SERVICE" -f --output=short-precise ;;

    logs-tail)
        echo "Last 100 log lines:"
        journalctl -u "$SERVICE" -n 100 --no-pager ;;

    knowledge)
        echo "Knowledge bank files:"
        ls -lh "$BOT_DIR/knowledge/" 2>/dev/null || echo "No knowledge folder found"
        echo ""
        echo "Total size:"
        du -sh "$BOT_DIR/knowledge/" 2>/dev/null ;;

    update)
        echo "Updating Python packages..."
        "$BOT_DIR/venv/bin/pip" install -r "$BOT_DIR/requirements.txt" -q
        echo "Restarting bot..."
        systemctl restart "$SERVICE"
        systemctl status "$SERVICE" --no-pager ;;

    memory)
        echo "VPS memory:"
        free -h
        echo ""
        echo "Bot process:"
        ps aux | grep bot.py | grep -v grep || echo "(bot not running as process)" ;;

    *)
        echo ""
        echo "Discord AI Bot management commands"
        echo "  bash manage.sh start       Start the bot"
        echo "  bash manage.sh stop        Stop the bot"
        echo "  bash manage.sh restart     Restart the bot"
        echo "  bash manage.sh status      Check if bot is running"
        echo "  bash manage.sh logs        Stream live logs"
        echo "  bash manage.sh logs-tail   Show last 100 log lines"
        echo "  bash manage.sh knowledge   List knowledge bank files"
        echo "  bash manage.sh update      Update packages and restart"
        echo "  bash manage.sh memory      Show VPS memory usage"
        echo "" ;;
esac
