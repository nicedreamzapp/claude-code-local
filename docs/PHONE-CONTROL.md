# 📱 Control From Your Phone

You don't have to be at your Mac to use this. Text a command, get back a full video.

```
📱 Your iPhone                    💻 Your Mac
     │                                │
     │── "find me an article  ──────>│── imessage-receive.sh reads it
     │    and send me a video"        │── local model plans the task
     │                                │── Brave browser finds the article
     │                                │── speak narrates in your voice
     │                                │── Studio Record captures it all
     │<── 🎥 video in iMessage ──────│── imessage-send-video.sh ships it
     │                                │
   🛋️  From your couch            🖥️  At your desk
```

**Everything works — text, images, and video:**

| Command | What happens | You get |
|---|---|---|
| "summarize this article" | Local model reads + replies | 💬 Text |
| "send me a screenshot of X" | Claude screenshots | 📸 Image in iMessage |
| "screen record you doing Y" | Records + sends | 🎥 Video in iMessage |
| "make me a produced video" | Full edit pipeline | 🎬 Title card + subs |

**Full pipeline repo:** [nicedreamzapp/claude-screen-to-phone](https://github.com/nicedreamzapp/claude-screen-to-phone)
→ Clone it, run `setup.sh`, fill in your phone number. Works with this local AI stack or Claude cloud. Docs in [IMESSAGE_MEDIA_PIPELINE.md](IMESSAGE_MEDIA_PIPELINE.md).

---

[← back to the README](../README.md)
