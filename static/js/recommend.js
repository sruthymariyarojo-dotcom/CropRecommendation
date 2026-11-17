
/* ============================================================
   CROPWISE - CROP RECOMMENDATION PAGE INTERACTIONS
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    "use strict";


    // --------------------------------------------------------
    // 1. GET PAGE ELEMENTS
    // --------------------------------------------------------

    const form = document.getElementById("recommendForm");

    const exampleButton = document.getElementById("loadExampleButton");

    const predictButton = document.getElementById("predictButton");

    const formStatus = document.getElementById("formStatus");


    // --------------------------------------------------------
    // 2. VERIFIED EXAMPLE INPUT VALUES
    // --------------------------------------------------------

    const exampleValues = {

        N: "13",

        P: "23",

        K: "6",

        temperature: "23.961476",

        humidity: "90.26408",

        ph: "7.365338",

        rainfall: "102.69587"

    };


    // --------------------------------------------------------
    // 3. LOAD EXAMPLE VALUES
    // --------------------------------------------------------

    if (exampleButton && form) {

        exampleButton.addEventListener("click", () => {

            Object.entries(exampleValues).forEach(
                ([name, value]) => {

                    const field = form.elements.namedItem(name);

                    if (field) {

                        field.value = value;

                        field.dispatchEvent(
                            new Event("input", {
                                bubbles: true
                            })
                        );

                    }

                }
            );


            // Display confirmation message.
            if (formStatus) {

                formStatus.textContent =
                    "Verified example values loaded. Review them and select Recommend crop.";

                formStatus.classList.add("is-visible");

            }


            // Focus the first input field.
            const firstField = form.elements.namedItem("N");

            if (firstField) {

                firstField.focus({
                    preventScroll: true
                });

            }

        });

    }



    // --------------------------------------------------------
    // 4. HANDLE FORM INTERACTIONS
    // --------------------------------------------------------

    if (form) {


        // Remove field-specific errors when the user edits a field.

        form.querySelectorAll(".rec-field input").forEach(
            (field) => {

                field.addEventListener("input", () => {

                    const wrap = field.closest(".rec-input-wrap");

                    const container = field.closest(".rec-field");


                    if (wrap) {

                        wrap.classList.remove("has-error");

                    }


                    if (container) {

                        const errorMessage =
                            container.querySelector(".rec-field-error");

                        if (errorMessage) {

                            errorMessage.remove();

                        }

                    }

                });

            }
        );



        // ----------------------------------------------------
        // 5. FORM SUBMISSION
        // ----------------------------------------------------

        form.addEventListener("submit", (event) => {


            // Check browser-level validation.
            if (!form.checkValidity()) {

                event.preventDefault();

                form.reportValidity();

                return;

            }


            // Show loading state.
            if (predictButton) {

                predictButton.classList.add("is-loading");

                predictButton.disabled = true;

                predictButton.setAttribute(
                    "aria-busy",
                    "true"
                );

            }


            if (formStatus) {

                formStatus.textContent =
                    "Analysing your values with the trained model…";

                formStatus.classList.add("is-visible");

            }

        });



        // ----------------------------------------------------
        // 6. RESET FORM
        // ----------------------------------------------------

        form.addEventListener("reset", () => {

            window.setTimeout(() => {


                // Remove validation error styling.
                form.querySelectorAll(
                    ".rec-input-wrap.has-error"
                ).forEach((wrap) => {

                    wrap.classList.remove("has-error");

                });


                // Remove field error messages.
                form.querySelectorAll(
                    ".rec-field-error"
                ).forEach((message) => {

                    message.remove();

                });


                if (formStatus) {

                    formStatus.textContent =
                        "The form has been cleared.";

                    formStatus.classList.add("is-visible");

                }


                // Restore prediction button.
                if (predictButton) {

                    predictButton.classList.remove("is-loading");

                    predictButton.disabled = false;

                    predictButton.removeAttribute("aria-busy");

                }

            }, 0);

        });

    }

    // Set dynamic probability bar widths
    document.querySelectorAll(".rec-alternative-fill").forEach((bar) => {

        const probability = Number(bar.dataset.probability);

        if (Number.isFinite(probability)) {

            const safeWidth = Math.min(100, Math.max(0, probability));

            requestAnimationFrame(() => {
                bar.style.width = `${safeWidth}%`;
            });

        }

    });



    // --------------------------------------------------------
    // 7. SCROLL TO PREDICTION RESULT
    // --------------------------------------------------------

    // The result is now positioned below the entire input form.
    // After Flask returns the prediction, scroll to that section.

    const resultOrError = document.querySelector(
        ".rec-result, .rec-alert-error"
    );


    if (resultOrError) {

        resultOrError.setAttribute("tabindex", "-1");


        window.requestAnimationFrame(() => {

            resultOrError.scrollIntoView({

                behavior: "smooth",

                block: "start"

            });

            resultOrError.focus({
                preventScroll: true
            });

        });

    }

});
