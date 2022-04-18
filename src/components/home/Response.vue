<template>
  <v-card-title light class="title">
    <v-dialog v-model="dismiss_dialog" persistent max-width="500">
      <template v-slot:activator="{ on, attrs }">
        <v-btn light v-bind="attrs" v-on="on">
          <v-icon>mdi-check</v-icon>Dismiss
        </v-btn>
      </template>
      <v-card>
        <v-card-title class="headline">Are you sure you want to dismiss?</v-card-title>
        <v-card-text>Dismiss only if you HAVE sent help</v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="error" outlined @click="dismiss_dialog = false">Cancel</v-btn>
          <v-btn color="green darken-1" outlined @click="dismiss">Dismiss</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-spacer></v-spacer>
    <v-dialog v-model="report_dialog" persistent max-width="500">
      <template v-slot:activator="{ on, attrs }">
        <v-btn dark v-bind="attrs" v-on="on" text>Not an accident?</v-btn>
      </template>
      <v-card>
        <v-card-title class="headline">Do you want to report this image as NOT an Accident?</v-card-title>
        <v-card-text>Reporting images removes it from the database and helps us improve our Accident Detection accuracy</v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="error" outlined @click="report_dialog = false">Cancel</v-btn>
          <v-btn color="green darken-1" outlined @click="report">Report</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-card-title>
</template>
      

<script>
import db from "@/firebase/init";
// import firebase from "firebase";
export default {
  name: "Response",
  props: ["doc_id"],
  data() {
    return {
      dismiss_dialog: false,
      report_dialog: false
    };
  },
  methods: {
    dismiss() {
      var docRef = db.collection("accidents").doc(this.doc_id);
      docRef
        .get()
        .then(function(doc) {
          if (doc.exists) {
            docRef.update({
              is_dismissed: true
            });
          } else {
            // doc.data() will be undefined in this case
            console.log("No such document!");
          }
        })
        .catch(function(error) {
          console.log("Error getting document:", error);
        });
    },
    report() {
      var docRef = db.collection("accidents").doc(this.doc_id);
      docRef
        .get()
        .then(function(doc) {
          if (doc.exists) {
            docRef.update({
              is_reported: true
            });
          } else {
            // doc.data() will be undefined in this case
            console.log("No such document!");
          }
        })
        .catch(function(error) {
          console.log("Error getting document:", error);
        });
    }
  }
};
</script>


