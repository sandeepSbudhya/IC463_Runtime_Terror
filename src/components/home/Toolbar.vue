<template>
  <v-card color="grey lighten-4" flat height="64px" tile>
    <v-app-bar fixed app color="primary" dark>
      <v-app-bar-nav-icon @click.stop="$emit('toggle-drawer')" v-if="this.$root.authenticated"></v-app-bar-nav-icon>

      <v-toolbar-title>Accident Detection</v-toolbar-title>

      <v-spacer></v-spacer>

      <span v-if="!this.$root.authenticated">
        <v-btn
          light
          class="ma-1"
          v-for="button in buttons"
          :key="button.title"
          router
          :to="button.route"
        >
          <span>{{ button.title }}</span>
          <v-icon>{{ button.icon }}</v-icon>
        </v-btn>
      </span>

      <span>
        <v-btn @click="logout()" v-if="this.$root.authenticated" light>
          <span>Logout</span>
          <v-icon>mdi-logout</v-icon>
        </v-btn>
      </span>
    </v-app-bar>
  </v-card>
</template>

<script>
import firebase from "firebase";
export default {
  name: "Toolbar",
  components: {},

  data: () => ({
    buttons: [
      { title: "Login", icon: "mdi-login", route: "/login" },
      { title: "Signup", icon: "mdi-signup", route: "/signup" }
    ]
  }),
  methods: {
    logout() {
      this.$root.authenticated = false;
      console.log(this.$root.authenticated);
      firebase
        .auth()
        .signOut()
        .then(() => {
          this.$router.push({ name: "Login" });
        });
    }
  }
};
</script>
