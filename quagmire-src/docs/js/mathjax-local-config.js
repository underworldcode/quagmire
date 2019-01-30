/* Set local configuration for mathjax */

/* Set local configuration for mathjax */

// MathJax.Hub.Config({
//     extensions: ["tex2jax.js"],
//     jax: ["input/TeX", "output/HTML-CSS"],
//     tex2jax: {
//       inlineMath: [ ['$','$'], ["\\(","\\)"] ],
//       displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
//       processEscapes: true
//     },
//     "HTML-CSS": { availableFonts: ["TeX"] }
//   });
//
// /* Add line numbers by default */
//
// MathJax.Hub.Config({
//   TeX: { equationNumbers: { autoNumber: "AMS" } }
// });


/* mathjax-loader.js  file */
/* ref: http://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/ */
(function (win, doc) {
  win.MathJax = {
    config: ["MMLorHTML.js"],
    extensions: ["tex2jax.js"],
    jax: ["input/TeX"],
    tex2jax: {
      inlineMath: [ ["\\(","\\)"] ],
      displayMath: [ ["\\[","\\]"] ]
    },
    TeX: {
      TagSide: "left",
      TagIndent: "1cm",
      MultLineWidth: "85%",
      equationNumbers: {
        autoNumber: "AMS",
      },
      unicode: {
        fonts: "STIXGeneral,'Arial Unicode MS'"
      }
    },
    displayAlign: 'left',
    showProcessingMessages: false,
    messageStyle: 'none'
  };
})(window, document);
