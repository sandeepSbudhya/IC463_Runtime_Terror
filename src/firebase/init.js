import firebase from 'firebase'
// import firestore from 'firebase/firestore'
var firebaseConfig = {
  apiKey: "AIzaSyBVsJbZO4MaT4wrsmUw98t_HCyZwEE-EsQ",
  authDomain: "accident-detection-54513.firebaseapp.com",
  databaseURL: "https://accident-detection-54513.firebaseio.com",
  projectId: "accident-detection-54513",
  storageBucket: "accident-detection-54513.appspot.com",
  messagingSenderId: "115519272790",
  appId: "1:115519272790:web:35ad75e3851d9492be48f7"
};


  // Initialize Firebase
  const firebaseApp = firebase.initializeApp(firebaseConfig);
  export default firebaseApp.firestore()