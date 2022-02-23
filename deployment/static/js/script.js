"use strict";

const container = document.getElementById("take_container");
const container_next = document.getElementById("take_container_next");
const username = document.querySelector(".username");
const subject_selected = document.getElementById("subject_select");
const button = new Array();
const name_user = username.textContent;
const data = {
  SS: {
    name: "Siddhant Sharma",
    subject: ["Big Data", "Information System"],
  },
  PG: {
    name: "Punam Gwachha",
    subject: ["Multimedia System"],
  },
  AK: {
    name: "Abhijit Karna",
    subject: ["Simulation and Modelling"],
  },
  PS: {
    name: "Prabesh Shrestha",
    subject: ["Internet and Intranet"],
  },
  RP: {
    name: "Rabindra Phonju",
    subject: ["Engineering Professional Practise"],
  },
  SP: {
    name: "Sajit Pyakurel",
    subject: ["Information System"],
  },
};
const { subject } = data[name_user];

const button_fun = function (sub) {
  var button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-primary btn-lg subbutton";
  button.id = sub;
  button.innerHTML = sub;
  container.appendChild(button);
};

for (let index = 0; index < subject.length; index++) {
  const element = subject[index];
  button_fun(element);
  button[index] = document.getElementById(element);
}

for (let index = 0; index < button.length; index++) {
  button[index].addEventListener("click", function () {
    subject_selected.value = button[index].innerHTML;
    container_next.classList.remove("hidden");
    container.classList.add("hidden");
  });
}
