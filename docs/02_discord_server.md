# Discord Server Setup Notes

## Discord Developer Portal

1. Create an application.
2. Add a bot user.
3. Enable the Message Content Intent.
4. Invite the bot with permissions to read messages, send messages, and use embeds if you add richer output later.

The bot does not require the privileged Members intent.

## Channel Allowlist

Use `ALLOWED_CHANNEL_IDS` for real servers because channel names can change. Keep `ALLOWED_CHANNELS` only for demos or local testing.
