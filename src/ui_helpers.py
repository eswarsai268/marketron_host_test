import streamlit as st
import time

def stream_text(text):
    """Takes fully generated text and visually streams it word-by-word."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) # Speed of the typing effect

def scroll_to_bottom():
    st.html("""
        <script>
            (function() {
                function scrollEl(el, tag) {
                    if (!el) { console.log('[scroll-debug] ' + tag + ': element not found'); return; }
                    console.log('[scroll-debug] ' + tag + ': scrollHeight=' + el.scrollHeight + ' clientHeight=' + el.clientHeight + ' scrollTop(before)=' + el.scrollTop);
                    try {
                        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                        console.log('[scroll-debug] ' + tag + ': scrollTo called, scrollTop(after)=' + el.scrollTop);
                    } catch (e) {
                        console.log('[scroll-debug] ' + tag + ': scrollTo threw, falling back. Error: ' + e);
                        el.scrollTop = el.scrollHeight;
                    }
                }

                function findScrollable() {
                    var container = document.querySelector('.st-key-campaign_chat_box');
                    console.log('[scroll-debug] container found: ' + (container ? 'yes' : 'NO'));
                    if (!container) return null;
                    if (container.scrollHeight > container.clientHeight) {
                        console.log('[scroll-debug] container itself is scrollable');
                        return container;
                    }
                    var descendants = container.querySelectorAll('*');
                    console.log('[scroll-debug] checking ' + descendants.length + ' descendants for overflow');
                    for (var i = 0; i < descendants.length; i++) {
                        if (descendants[i].scrollHeight > descendants[i].clientHeight) {
                            console.log('[scroll-debug] found scrollable descendant: ' + descendants[i].tagName + '.' + descendants[i].className);
                            return descendants[i];
                        }
                    }
                    console.log('[scroll-debug] no scrollable descendant found, using container as fallback');
                    return container;
                }

                console.log('[scroll-debug] scroll_to_bottom() invoked at ' + Date.now());
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        scrollEl(findScrollable(), 'main-call');
                    });
                });
            })();
        </script>
    """, unsafe_allow_javascript=True)

def preserve_scroll():
    st.html("""
        <script>
            (function() {
                var mainEl = document.querySelector('section[data-testid="stMain"]');
                if (mainEl) {
                    var saved = sessionStorage.getItem('scrollPos');
                    if (saved !== null) { mainEl.scrollTop = parseInt(saved); }
                    mainEl.addEventListener('scroll', function() {
                        sessionStorage.setItem('scrollPos', mainEl.scrollTop);
                    });
                }
            })();
        </script>
    """, unsafe_allow_javascript=True)